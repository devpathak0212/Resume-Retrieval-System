import pandas as pd
import numpy as np
import faiss
import json

MODELS = ['minilm', 'mpnet', 'bge']
TOP_N_CHUNKS = 30
TOP_K_RESUMES = 5

def build_faiss_index(chunk_embeddings):
    dim = chunk_embeddings.shape[1]
    # Using IndexFlatIP (inner product) since embeddings are normalized -> equivalent to cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(chunk_embeddings.astype('float32'))
    return index

def retrieve_top_k_resumes(index, job_embedding, chunk_metadata, top_n_chunks, top_k_resumes):
    job_embedding = job_embedding.astype('float32').reshape(1, -1)
    scores, indices = index.search(job_embedding, top_n_chunks)
    scores = scores[0]
    indices = indices[0]

    # Aggregate by resume_id via sum of chunk scores
    resume_scores = {}
    for score, idx in zip(scores, indices):
        resume_id = chunk_metadata.iloc[idx]['resume_id']
        resume_scores[resume_id] = resume_scores.get(resume_id, 0.0) + float(score)

    # Sort resumes by aggregated score, descending
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
            ranked = retrieve_top_k_resumes(index, job_emb[job_idx], chunks, TOP_N_CHUNKS, TOP_K_RESUMES)
            unique_resume_counts.append(len(ranked))
            model_results[category] = [
                {'resume_id': int(rid), 'score': score} for rid, score in ranked
            ]

        all_results[model_name] = model_results

        print(f"Min unique resumes returned across all 24 jobs: {min(unique_resume_counts)}")
        print(f"Max unique resumes returned across all 24 jobs: {max(unique_resume_counts)}")
        print(f"Jobs with fewer than {TOP_K_RESUMES} unique resumes: {sum(1 for c in unique_resume_counts if c < TOP_K_RESUMES)}")

    with open('retrieval_results_cosine.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\nSaved retrieval_results_cosine.json")

if __name__ == "__main__":
    main()