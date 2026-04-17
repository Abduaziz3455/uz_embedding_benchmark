#!/usr/bin/env python3
"""
Hard Negative Analysis — measures how well a model distinguishes true
positives from hard negatives (topically similar but incorrect passages).

Requires a dataset with hard_negative_ids (the shipped uz_news_benchmark.json has them).

Metrics:
  - Triplet Accuracy       — sim(q, pos) > sim(q, neg) per pair
  - Discrimination Rate    — pos ranked above ALL hard negatives per query
  - Avg Margin             — mean of sim(q, pos) - max(sim(q, hard_neg))
  - Restricted MRR         — MRR over only {pos ∪ hard_negatives}
  - Positive / HN Avg Rank — corpus ranks for context
"""

import argparse
import json
import os
import sys

from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark import (
    DEFAULT_DATASET,
    cosine_similarity_matrix,
    load_dataset,
    rank_passages,
    run_gemini_benchmark,
    run_local_benchmark,
)

console = Console()


def triplet_accuracy(sim_matrix, query_idx, corpus_idx, queries):
    correct = total = 0
    for q in queries:
        qi = query_idx[q["id"]]
        for pos_id in q["relevant_ids"]:
            pos_sim = sim_matrix[qi, corpus_idx[pos_id]]
            for neg_id in q.get("hard_negative_ids", []):
                neg_sim = sim_matrix[qi, corpus_idx[neg_id]]
                if pos_sim > neg_sim:
                    correct += 1
                total += 1
    return correct / total if total else 0.0


def discrimination_rate(sim_matrix, query_idx, corpus_idx, queries):
    wins = total = 0
    for q in queries:
        hard_negs = q.get("hard_negative_ids", [])
        if not hard_negs:
            continue
        qi = query_idx[q["id"]]
        pos_sim = max(sim_matrix[qi, corpus_idx[pid]] for pid in q["relevant_ids"])
        max_neg_sim = max(sim_matrix[qi, corpus_idx[nid]] for nid in hard_negs)
        if pos_sim > max_neg_sim:
            wins += 1
        total += 1
    return wins / total if total else 0.0


def avg_margin(sim_matrix, query_idx, corpus_idx, queries):
    margins = []
    for q in queries:
        hard_negs = q.get("hard_negative_ids", [])
        if not hard_negs:
            continue
        qi = query_idx[q["id"]]
        pos_sim = max(sim_matrix[qi, corpus_idx[pid]] for pid in q["relevant_ids"])
        max_neg_sim = max(sim_matrix[qi, corpus_idx[nid]] for nid in hard_negs)
        margins.append(float(pos_sim - max_neg_sim))
    return sum(margins) / len(margins) if margins else 0.0


def restricted_mrr(sim_matrix, query_idx, corpus_idx, queries):
    rrs = []
    for q in queries:
        hard_negs = q.get("hard_negative_ids", [])
        if not hard_negs:
            continue
        qi = query_idx[q["id"]]
        rel_set = set(q["relevant_ids"])
        candidate_ids = list(q["relevant_ids"]) + hard_negs
        scores = [(cid, float(sim_matrix[qi, corpus_idx[cid]])) for cid in candidate_ids]
        scores.sort(key=lambda x: -x[1])
        rr = 0.0
        for rank, (cid, _) in enumerate(scores, 1):
            if cid in rel_set:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    return sum(rrs) / len(rrs) if rrs else 0.0


def hard_neg_avg_rank(rankings, queries):
    ranks = []
    for q in queries:
        hard_negs = q.get("hard_negative_ids", [])
        if not hard_negs:
            continue
        ranked = rankings[q["id"]]
        neg_set = set(hard_negs)
        for rank, doc_id in enumerate(ranked, 1):
            if doc_id in neg_set:
                ranks.append(rank)
                break
    return sum(ranks) / len(ranks) if ranks else 0.0


def positive_avg_rank(rankings, queries):
    ranks = []
    for q in queries:
        rel_set = set(q["relevant_ids"])
        ranked = rankings[q["id"]]
        for rank, doc_id in enumerate(ranked, 1):
            if doc_id in rel_set:
                ranks.append(rank)
                break
    return sum(ranks) / len(ranks) if ranks else 0.0


def compute_hard_negative_metrics(sim_matrix, rankings, query_idx, corpus_idx, queries):
    return {
        "triplet_accuracy": triplet_accuracy(sim_matrix, query_idx, corpus_idx, queries),
        "discrimination_rate": discrimination_rate(sim_matrix, query_idx, corpus_idx, queries),
        "avg_margin": avg_margin(sim_matrix, query_idx, corpus_idx, queries),
        "restricted_mrr": restricted_mrr(sim_matrix, query_idx, corpus_idx, queries),
        "positive_avg_rank": positive_avg_rank(rankings, queries),
        "hard_neg_avg_rank": hard_neg_avg_rank(rankings, queries),
    }


def main():
    parser = argparse.ArgumentParser(description="Hard Negative Analysis for Uzbek Embedding Benchmark")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--local", action="store_true", help="Use sentence-transformers (local)")
    source.add_argument("--gemini", action="store_true", help="Use Gemini embedding API")

    parser.add_argument("--model-name", help="Model name")
    parser.add_argument("--gemini-api-key", help="Gemini API key (or set GEMINI_API_KEY env)")
    parser.add_argument("--gemini-model", default="gemini-embedding-001")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--passage-prefix", default="")
    parser.add_argument("--query-prompt-name", default=None)
    parser.add_argument("--passage-prompt-name", default=None)
    parser.add_argument("--st-task", default=None)
    parser.add_argument("--transliterate", default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", help="Output JSON path (auto-generated if not set)")
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    dataset = load_dataset(args.dataset)

    has_hn = sum(1 for q in dataset["queries"] if q.get("hard_negative_ids"))
    if has_hn == 0:
        console.print("[red]Dataset has no hard_negative_ids. Use uz_news_benchmark.json.[/red]")
        sys.exit(1)

    console.print("[bold green]Hard Negative Analysis[/bold green]")
    console.print(f"  Dataset: {args.dataset}")
    console.print(f"  Corpus: {len(dataset['corpus'])} passages")
    console.print(f"  Queries: {len(dataset['queries'])} ({has_hn} with hard negatives)")

    if args.gemini:
        api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            console.print("[red]Gemini API key required[/red]")
            sys.exit(1)
        result = run_gemini_benchmark(api_key, args.gemini_model, dataset, args.batch_size)
    else:
        if not args.model_name:
            console.print("[red]--model-name required[/red]")
            sys.exit(1)
        result = run_local_benchmark(
            args.model_name, dataset, args.batch_size,
            args.query_prefix, args.passage_prefix, args.trust_remote_code,
            device=args.device,
            query_prompt_name=args.query_prompt_name,
            passage_prompt_name=args.passage_prompt_name,
            st_task=args.st_task,
            transliterate_mode=args.transliterate,
            max_seq_length=args.max_seq_length,
        )

    console.print("\n[bold]Computing similarities and rankings...[/bold]")
    sim_matrix = cosine_similarity_matrix(result["query_embs"], result["corpus_embs"])
    rankings = rank_passages(sim_matrix, result["query_ids"], result["corpus_ids"])

    query_idx = {qid: i for i, qid in enumerate(result["query_ids"])}
    corpus_idx = {cid: i for i, cid in enumerate(result["corpus_ids"])}

    console.print("[bold]Computing hard negative metrics...[/bold]")
    hn_metrics = compute_hard_negative_metrics(
        sim_matrix, rankings, query_idx, corpus_idx, dataset["queries"],
    )

    console.print(f"\n[bold green]Results: {result['model_name']}[/bold green]")
    console.print(f"  Embedding dim: {result['embedding_dim']}")

    hn_table = Table(title="Hard Negative Metrics")
    hn_table.add_column("Metric", style="cyan")
    hn_table.add_column("Value", style="bold white", justify="right")
    hn_table.add_column("Description", style="dim")

    hn_table.add_row("Triplet Accuracy", f"{hn_metrics['triplet_accuracy']:.4f}",
                     "% of (pos, neg) pairs ranked correctly")
    hn_table.add_row("Discrimination Rate", f"{hn_metrics['discrimination_rate']:.4f}",
                     "% of queries: pos above ALL hard negs")
    hn_table.add_row("Avg Margin", f"{hn_metrics['avg_margin']:.4f}",
                     "sim(q,pos) - max(sim(q,neg)), higher=better")
    hn_table.add_row("Restricted MRR", f"{hn_metrics['restricted_mrr']:.4f}",
                     "MRR over {pos ∪ hard_negs} only")
    hn_table.add_row("Positive Avg Rank", f"{hn_metrics['positive_avg_rank']:.1f}",
                     "avg corpus rank of true positive")
    hn_table.add_row("Hard Neg Avg Rank", f"{hn_metrics['hard_neg_avg_rank']:.1f}",
                     "avg corpus rank of top hard negative")
    console.print(hn_table)

    output_data = {
        "model_name": result["model_name"],
        "embedding_dim": result["embedding_dim"],
        "dataset": args.dataset,
        "num_corpus": len(dataset["corpus"]),
        "num_queries": len(dataset["queries"]),
        "num_with_hard_negatives": has_hn,
        "hard_negative_metrics": hn_metrics,
    }

    if args.output:
        output_path = args.output
    else:
        os.makedirs("results_news", exist_ok=True)
        safe_name = result["model_name"].replace("/", "_").replace(" ", "_")
        output_path = f"results_news/{safe_name}_hard_neg.json"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    console.print(f"\n[dim]Results saved to {output_path}[/dim]")


if __name__ == "__main__":
    main()
