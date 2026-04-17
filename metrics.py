"""Retrieval evaluation metrics for embedding benchmark."""

import math
from typing import Dict, List


def hit_rate_at_k(
    rankings: Dict[str, List[str]],
    relevants: Dict[str, List[str]],
    k: int,
) -> float:
    """Fraction of queries where at least one relevant doc appears in top-K."""
    hits = 0
    for qid, ranked_ids in rankings.items():
        rel_set = set(relevants.get(qid, []))
        if rel_set & set(ranked_ids[:k]):
            hits += 1
    return hits / len(rankings) if rankings else 0.0


def recall_at_k(
    rankings: Dict[str, List[str]],
    relevants: Dict[str, List[str]],
    k: int,
) -> float:
    """Average fraction of relevant docs found in top-K across all queries."""
    scores = []
    for qid, ranked_ids in rankings.items():
        rel_set = set(relevants.get(qid, []))
        if not rel_set:
            continue
        found = len(rel_set & set(ranked_ids[:k]))
        scores.append(found / len(rel_set))
    return sum(scores) / len(scores) if scores else 0.0


def mrr(
    rankings: Dict[str, List[str]],
    relevants: Dict[str, List[str]],
) -> float:
    """Mean Reciprocal Rank — average of 1/rank of first relevant result."""
    rrs = []
    for qid, ranked_ids in rankings.items():
        rel_set = set(relevants.get(qid, []))
        rr = 0.0
        for rank, doc_id in enumerate(ranked_ids, 1):
            if doc_id in rel_set:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    return sum(rrs) / len(rrs) if rrs else 0.0


def ndcg_at_k(
    rankings: Dict[str, List[str]],
    relevants: Dict[str, List[str]],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at K (binary relevance)."""
    scores = []
    for qid, ranked_ids in rankings.items():
        rel_set = set(relevants.get(qid, []))
        if not rel_set:
            continue

        # DCG: sum of 1/log2(i+2) for relevant docs in top-K
        dcg = 0.0
        for i, doc_id in enumerate(ranked_ids[:k]):
            if doc_id in rel_set:
                dcg += 1.0 / math.log2(i + 2)

        # Ideal DCG: all relevant docs at the top
        ideal_hits = min(len(rel_set), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def compute_all_metrics(
    rankings: Dict[str, List[str]],
    relevants: Dict[str, List[str]],
    k_values: List[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    """Compute all metrics at multiple K values. Returns flat dict."""
    results = {"mrr": mrr(rankings, relevants)}
    for k in k_values:
        results[f"hit_rate@{k}"] = hit_rate_at_k(rankings, relevants, k)
        results[f"recall@{k}"] = recall_at_k(rankings, relevants, k)
        results[f"ndcg@{k}"] = ndcg_at_k(rankings, relevants, k)
    return results
