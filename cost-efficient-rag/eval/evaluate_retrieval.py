# ============================================================
# eval/evaluate_retrieval.py
#
# WHAT THIS FILE DOES:
# Loads data/eval_dataset.json (your 15-30 gold questions), runs
# EACH question through the vector store's search(), and computes
# standard "information retrieval" quality metrics:
#   - Recall@k / Hit Rate
#   - Mean Reciprocal Rank (MRR)
#   - nDCG@k
#
# RUN IT WITH:
#   python -m eval.evaluate_retrieval
# ============================================================

import json
import math
from typing import List

import numpy as np

from src.vector_store import VectorStoreManager

TOP_K = 5  # how many chunks we retrieve per question


# ------------------------------------------------------------
# Metric functions — each takes the ranked list of retrieved chunk
# IDs and the gold list of relevant chunk IDs for ONE question.
# ------------------------------------------------------------

def compute_recall_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
    """
    Fraction of the gold-relevant chunks that we managed to retrieve
    in our top k results. 1.0 = found all of them, 0.0 = found none.
    """
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_set = set(gold_ids)
    if not relevant_set:
        return 0.0
    hits = len(top_k_retrieved.intersection(relevant_set))
    return hits / len(relevant_set)


def compute_hit_rate(retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
    """
    1.0 if AT LEAST ONE relevant chunk appears in the top k, else 0.0.
    This is a softer/simpler metric than Recall@k.
    """
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_set = set(gold_ids)
    return 1.0 if top_k_retrieved.intersection(relevant_set) else 0.0


def compute_mrr(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    """
    1 / (rank of the FIRST relevant chunk we retrieved).
    If the first relevant chunk is in position 1 -> score 1.0.
    If it's in position 4 -> score 0.25. If none found -> 0.0.
    """
    relevant_set = set(gold_ids)
    for index, item_id in enumerate(retrieved_ids):
        if item_id in relevant_set:
            return 1.0 / (index + 1)
    return 0.0


def compute_ndcg_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain.
    Rewards relevant chunks MORE when they appear EARLIER in the results.
    Here relevance is binary (1 if in gold_ids, else 0), which is the
    standard simplification when you don't have graded relevance scores.
    """
    relevant_set = set(gold_ids)
    top_k = retrieved_ids[:k]

    # DCG: sum of (relevance / log2(position + 1)) for each retrieved item.
    dcg = 0.0
    for i, item_id in enumerate(top_k):
        relevance = 1.0 if item_id in relevant_set else 0.0
        position = i + 1
        dcg += relevance / math.log2(position + 1)

    # IDCG: the BEST POSSIBLE dcg, i.e. if all relevant chunks were
    # ranked first. We use this to normalize DCG into a 0-1 range.
    ideal_relevances = [1.0] * min(len(relevant_set), k)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevances))

    if idcg == 0:
        return 0.0
    return dcg / idcg


# ------------------------------------------------------------
# Main evaluation loop
# ------------------------------------------------------------

def run_retrieval_evaluation():
    with open("data/eval_dataset.json", "r") as f:
        eval_questions = json.load(f)

    vector_store = VectorStoreManager()

    recall_scores, hit_scores, mrr_scores, ndcg_scores = [], [], [], []
    per_question_results = []

    for item in eval_questions:
        question = item["question"]
        gold_ids = item["relevant_chunk_ids"]

        results = vector_store.search(query_text=question, top_k=TOP_K)
        retrieved_ids = [r["id"] for r in results]

        recall = compute_recall_at_k(retrieved_ids, gold_ids, TOP_K)
        hit = compute_hit_rate(retrieved_ids, gold_ids, TOP_K)
        mrr = compute_mrr(retrieved_ids, gold_ids)
        ndcg = compute_ndcg_at_k(retrieved_ids, gold_ids, TOP_K)

        recall_scores.append(recall)
        hit_scores.append(hit)
        mrr_scores.append(mrr)
        ndcg_scores.append(ndcg)

        per_question_results.append({
            "question": question,
            "recall_at_k": recall,
            "hit_rate": hit,
            "mrr": mrr,
            "ndcg_at_k": ndcg,
        })

    summary = {
        "top_k": TOP_K,
        "num_questions": len(eval_questions),
        "avg_recall_at_k": round(float(np.mean(recall_scores)), 4),
        "avg_hit_rate": round(float(np.mean(hit_scores)), 4),
        "avg_mrr": round(float(np.mean(mrr_scores)), 4),
        "avg_ndcg_at_k": round(float(np.mean(ndcg_scores)), 4),
        "per_question": per_question_results,
    }

    print(json.dumps(summary, indent=2))

    with open("results/retrieval_eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    run_retrieval_evaluation()
