import json
import pandas as pd
import numpy as np

MODELS = ['minilm', 'mpnet', 'bge']
STRATEGIES = ['cosine', 'hybrid']
K = 5

def load_resume_to_category():
    chunks = pd.read_csv('..\\chunking\\resume_chunks.csv')
    return chunks.drop_duplicates('resume_id').set_index('resume_id')['Category'].to_dict()

def load_category_totals():
    # Total number of resumes per category, needed for Recall@K denominator
    chunks = pd.read_csv('..\\chunking\\resume_chunks.csv')
    unique_resumes = chunks.drop_duplicates('resume_id')
    return unique_resumes['Category'].value_counts().to_dict()

def precision_at_k(relevance_list):
    # relevance_list: list of 0/1, 1 = relevant (category matched)
    return sum(relevance_list) / len(relevance_list)

def recall_at_k(relevance_list, total_relevant):
    return sum(relevance_list) / total_relevant if total_relevant > 0 else 0.0

def mrr(relevance_list):
    for i, rel in enumerate(relevance_list):
        if rel == 1:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(relevance_list):
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_list))
    ideal_relevance = sorted(relevance_list, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
    return dcg / idcg if idcg > 0 else 0.0

def main():
    resume_to_category = load_resume_to_category()
    category_totals = load_category_totals()

    all_metrics = []

    for strategy in STRATEGIES:
        with open(f'..\\retrieval\\retrieval_results_{strategy}.json') as f:
            results = json.load(f)

        for model in MODELS:
            precisions, recalls, mrrs, ndcgs = [], [], [], []

            for job_category, retrieved in results[model].items():
                relevance_list = [
                    1 if resume_to_category.get(r['resume_id']) == job_category else 0
                    for r in retrieved
                ]
                total_relevant = category_totals.get(job_category, 0)

                precisions.append(precision_at_k(relevance_list))
                recalls.append(recall_at_k(relevance_list, total_relevant))
                mrrs.append(mrr(relevance_list))
                ndcgs.append(ndcg_at_k(relevance_list))

            all_metrics.append({
                'Strategy': strategy,
                'Model': model,
                f'Precision@{K}': np.mean(precisions),
                f'Recall@{K}': np.mean(recalls),
                'MRR': np.mean(mrrs),
                'NDCG': np.mean(ndcgs)
            })

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv('metrics_summary.csv', index=False)
    print(metrics_df.to_string(index=False))
    print("\nSaved metrics_summary.csv")

if __name__ == "__main__":
    main()