import pandas as pd
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():  # skip empty chunks
            chunks.append(chunk)
    return chunks

def main():
    df = pd.read_csv('..\\data\\Resume\\Resume.csv')

    # Drop empty resumes
    df = df[df['Resume_str'].str.split().str.len() > 0].copy()

    # Clean text
    df['cleaned_text'] = df['Resume_str'].apply(clean_text)

    # Build chunk-level records
    records = []
    for _, row in df.iterrows():
        chunks = chunk_text(row['cleaned_text'], chunk_size=200)
        for idx, chunk in enumerate(chunks):
            records.append({
                'resume_id': row['ID'],
                'chunk_index': idx,
                'total_chunks': len(chunks),
                'Category': row['Category'],
                'chunk_text': chunk
            })

    chunks_df = pd.DataFrame(records)
    chunks_df.to_csv('resume_chunks.csv', index=False)

    print("Saved resume_chunks.csv")
    print("Total resumes:", df.shape[0])
    print("Total chunks:", chunks_df.shape[0])
    print("Avg chunks per resume:", chunks_df.groupby('resume_id').size().mean())
    print("Max chunks for a single resume:", chunks_df.groupby('resume_id').size().max())
    print(chunks_df.head(5))

if __name__ == "__main__":
    main()