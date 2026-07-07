import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import time

def main():
    print("Loading MPNet model...")
    model = SentenceTransformer('all-mpnet-base-v2')

    print("Loading chunk data...")
    chunks = pd.read_csv('..\\chunking\\resume_chunks.csv')
    jobs = pd.read_csv('..\\job_descriptions\\job_descriptions.csv')

    print(f"Embedding {len(chunks)} chunks...")
    start = time.time()
    chunk_embeddings = model.encode(
        chunks['chunk_text'].tolist(),
        show_progress_bar=True,
        batch_size=32
    )
    print(f"Chunk embedding took {time.time() - start:.1f} seconds")

    print(f"Embedding {len(jobs)} job descriptions...")
    job_embeddings = model.encode(
        jobs['JobDescription'].tolist(),
        show_progress_bar=True,
        batch_size=32
    )

    print("Chunk embeddings shape:", chunk_embeddings.shape)
    print("Job embeddings shape:", job_embeddings.shape)

    np.save('chunk_embeddings_mpnet.npy', chunk_embeddings)
    np.save('job_embeddings_mpnet.npy', job_embeddings)
    print("Saved embeddings.")

if __name__ == "__main__":
    main()