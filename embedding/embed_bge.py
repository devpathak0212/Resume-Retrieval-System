import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import time

def main():
    print("Loading BGE model...")
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')

    print("Loading chunk data...")
    chunks = pd.read_csv('..\\chunking\\resume_chunks.csv')
    jobs = pd.read_csv('..\\job_descriptions\\job_descriptions.csv')

    # BGE-specific: queries (job descriptions) get instruction prefix; documents (chunks) do not
    query_prefix = "Represent this sentence for searching relevant passages: "
    prefixed_jobs = [query_prefix + jd for jd in jobs['JobDescription'].tolist()]

    print(f"Embedding {len(chunks)} chunks (no prefix, as documents)...")
    start = time.time()
    chunk_embeddings = model.encode(
        chunks['chunk_text'].tolist(),
        show_progress_bar=True,
        batch_size=32
    )
    print(f"Chunk embedding took {time.time() - start:.1f} seconds")

    print(f"Embedding {len(jobs)} job descriptions (with query prefix)...")
    job_embeddings = model.encode(
        prefixed_jobs,
        show_progress_bar=True,
        batch_size=32
    )

    print("Chunk embeddings shape:", chunk_embeddings.shape)
    print("Job embeddings shape:", job_embeddings.shape)

    np.save('chunk_embeddings_bge.npy', chunk_embeddings)
    np.save('job_embeddings_bge.npy', job_embeddings)
    print("Saved embeddings.")

if __name__ == "__main__":
    main()