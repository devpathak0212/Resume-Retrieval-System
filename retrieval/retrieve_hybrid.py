import pandas as pd
import numpy as np
import faiss
import json
import re

MODELS = ['minilm', 'mpnet', 'bge']
TOP_N_CHUNKS = 30
TOP_K_RESUMES = 5
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

STOPWORDS = set([
    'the', 'a', 'an', 'and', 'or', 'to', 'of', 'in', 'for', 'with', 'on', 'is', 'are',
    'as', 'by', 'this', 'that', 'be', 'will', 'preferred', 'required', 'experience',
    'role', 'roles', 'responsibilities', 'include', 'including', 'skills', 'strong',
    'seeking', 'professional'
])

def tokenize(text):
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return set(w for w in words if w not in STOPWORDS and len(w) > 2)

def keyword_overlap_score(job_words, chunk_text):
    chunk_words = tokenize(chunk_text)
    if not job_words:
        return 0.0
    overlap = job_words.intersection(chunk_words)
    return len(overlap) / len(job_words)

def normalize_scores(scores):
    scores = np.array(scores, dtype='float64')
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-9:  # avoid divide-by-zero if all scores are identical
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)

def build_faiss_index(chunk_embeddings):
    dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(chunk_embeddings.astype('float32'))
    return index

def retrieve_top_k_resumes_hybrid(index, job_embedding, job_text, chunk_metadata, top_n_chunks, top_k_resumes):
    job_embedding_arr = job_embedding.astype('float32').reshape(1, -1)

    # Retrieve a larger pool via semantic search first, then re-score with hybrid formula
    semantic_pool_size = 100
    sem_scores, indices = index.search(job_embedding_arr, semantic_pool_size)
    sem_scores = sem_scores[0]
    indices = indices[0]

    job_words = tokenize(job_text)

    # Compute raw keyword scores for the whole pool first
    kw_scores_raw = []
    for idx in indices:
        chunk_text = chunk_metadata.iloc[idx]['chunk_text']
        kw_scores_raw.append(keyword_overlap_score(job_words, chunk_text))

    # Normalize both score sets within this query's pool before combining
    # This ensures the 70/30 weighting is respected regardless of each
    # component's natural score range/spread
    sem_scores_norm = normalize_scores(sem_scores)
    kw_scores_norm = normalize_scores(kw_scores_raw)

    hybrid_scored = []
    for sem_n, kw_n, idx in zip(sem_scores_norm, kw_scores_norm, indices):
        hybrid_score = SEMANTIC_WEIGHT * float(sem_n) + KEYWORD_WEIGHT * float(kw_n)
        resume_id = chunk_metadata.iloc[idx]['resume_id']
        hybrid_scored.append((resume_id, hybrid_score))

    # Take top N chunks by hybrid score (not just semantic score) before aggregating
    hybrid_scored.sort(key=lambda x: x[1], reverse=True)
    top_chunks = hybrid_scored[:top_n_chunks]

    # Aggregate by resume_id via sum of hybrid scores
    resume_scores = {}
    for resume_id, score in top_chunks:
        resume_scores[resume_id] = resume_scores.get(resume_id, 0.0) + score

    ranked_resumes = sorted(resume_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_resumes[:top_k_resumes]

def main():
    chunks = pd.read_csv('..\\chunking\\resume_chunks.csv')
    jobs = pd.read_csv('..\\job_descriptions\\job_descriptions.csv')

    all_results = {}

    for model_name in MODELS:
        print(f"\n=== Processing {model_name} ===")
        chunk_emb = np.load(f'..\\embedding\\chunk_embeddings_{model_name}.npy')
        job_emb = np.load(f'..\\embedding\\job_embeddings_{model_name}.npy')

        print("Building FAISS index...")
        index = build_faiss_index(chunk_emb)

        model_results = {}
        unique_resume_counts = []

        for job_idx, job_row in jobs.iterrows():
            category = job_row['Category']
            job_text = job_row['JobDescription']
            ranked = retrieve_top_k_resumes_hybrid(
                index, job_emb[job_idx], job_text, chunks, TOP_N_CHUNKS, TOP_K_RESUMES
            )
            unique_resume_counts.append(len(ranked))
            model_results[category] = [
                {'resume_id': int(rid), 'score': score} for rid, score in ranked
            ]

        all_results[model_name] = model_results

        print(f"Min unique resumes returned across all 24 jobs: {min(unique_resume_counts)}")
        print(f"Max unique resumes returned across all 24 jobs: {max(unique_resume_counts)}")
        print(f"Jobs with fewer than {TOP_K_RESUMES} unique resumes: {sum(1 for c in unique_resume_counts if c < TOP_K_RESUMES)}")

    with open('retrieval_results_hybrid.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\nSaved retrieval_results_hybrid.json")

if __name__ == "__main__":
    main()