#!/usr/bin/env python3
"""
Uzbek Embedding Benchmark — standard retrieval metrics runner.

Most users should invoke this via `run.py`. Direct usage:

  # Local HuggingFace model (CUDA/MPS/CPU auto-detected):
  python benchmark.py --local --model-name BAAI/bge-m3

  # With query/passage prefixes (e5 family):
  python benchmark.py --local --model-name intfloat/multilingual-e5-large \
      --query-prefix "query: " --passage-prefix "passage: "

  # Gemini API:
  python benchmark.py --gemini
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from rich.console import Console
from rich.table import Table

from metrics import compute_all_metrics

console = Console()

DEFAULT_DATASET = "dataset/uz_news_benchmark.json"


def load_dataset(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def cosine_similarity_matrix(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    q_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-10)
    c_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-10)
    return q_norm @ c_norm.T


def rank_passages(
    sim_matrix: np.ndarray,
    query_ids: List[str],
    corpus_ids: List[str],
) -> Dict[str, List[str]]:
    rankings = {}
    for i, qid in enumerate(query_ids):
        scores = sim_matrix[i]
        sorted_indices = np.argsort(-scores)
        rankings[qid] = [corpus_ids[idx] for idx in sorted_indices]
    return rankings


def run_local_benchmark(
    model_name: str,
    dataset: dict,
    batch_size: int,
    query_prefix: str,
    passage_prefix: str,
    trust_remote_code: bool,
    device: str | None = None,
    query_prompt_name: str | None = None,
    passage_prompt_name: str | None = None,
    st_task: str | None = None,
    transliterate_mode: str | None = None,
    max_seq_length: int | None = None,
) -> dict:
    from models import SentenceTransformerClient

    client = SentenceTransformerClient(
        model_name, batch_size=batch_size, trust_remote_code=trust_remote_code,
        device=device, max_seq_length=max_seq_length,
    )
    console.print(f"  Device: {client.device}  max_seq_length: {client.max_seq_length}")

    corpus = dataset["corpus"]
    queries = dataset["queries"]
    corpus_ids = [p["id"] for p in corpus]
    query_ids = [q["id"] for q in queries]

    raw_corpus = [p["text"] for p in corpus]
    raw_queries = [q["text"] for q in queries]
    if transliterate_mode:
        from transliterate import apply as _translit_apply
        console.print(f"  Transliteration: {transliterate_mode}")
        raw_corpus = _translit_apply(transliterate_mode, raw_corpus)
        raw_queries = _translit_apply(transliterate_mode, raw_queries)

    corpus_texts = [f"{passage_prefix}{t}" for t in raw_corpus]
    query_texts = [f"{query_prefix}{t}" for t in raw_queries]

    console.print(f"\n[bold]Embedding {len(corpus_texts)} passages...[/bold]")
    corpus_embs, corpus_time = client.embed_timed(
        corpus_texts, prompt_name=passage_prompt_name, task=st_task
    )
    console.print(f"  Done in {corpus_time:.2f}s ({len(corpus_texts)/corpus_time:.1f} texts/s)")

    console.print(f"[bold]Embedding {len(query_texts)} queries...[/bold]")
    query_embs, query_time = client.embed_timed(
        query_texts, prompt_name=query_prompt_name, task=st_task
    )
    console.print(f"  Done in {query_time:.2f}s ({len(query_texts)/query_time:.1f} texts/s)")

    return {
        "corpus_embs": corpus_embs,
        "query_embs": query_embs,
        "corpus_ids": corpus_ids,
        "query_ids": query_ids,
        "corpus_time": corpus_time,
        "query_time": query_time,
        "model_name": client.name,
        "embedding_dim": corpus_embs.shape[1],
    }


def run_gemini_benchmark(
    api_key: str,
    model_name: str,
    dataset: dict,
    batch_size: int,
) -> dict:
    from models import GeminiEmbeddingClient

    client = GeminiEmbeddingClient(api_key, model_name, batch_size=batch_size)

    corpus = dataset["corpus"]
    queries = dataset["queries"]
    corpus_ids = [p["id"] for p in corpus]
    query_ids = [q["id"] for q in queries]

    corpus_texts = [p["text"] for p in corpus]
    query_texts = [q["text"] for q in queries]

    console.print(f"\n[bold]Embedding {len(corpus_texts)} passages (Gemini)...[/bold]")
    corpus_embs, corpus_time = client.embed_timed(corpus_texts, task_type="RETRIEVAL_DOCUMENT")
    console.print(f"  Done in {corpus_time:.2f}s ({len(corpus_texts)/corpus_time:.1f} texts/s)")

    console.print(f"[bold]Embedding {len(query_texts)} queries (Gemini)...[/bold]")
    query_embs, query_time = client.embed_timed(query_texts, task_type="RETRIEVAL_QUERY")
    console.print(f"  Done in {query_time:.2f}s ({len(query_texts)/query_time:.1f} texts/s)")

    return {
        "corpus_embs": corpus_embs,
        "query_embs": query_embs,
        "corpus_ids": corpus_ids,
        "query_ids": query_ids,
        "corpus_time": corpus_time,
        "query_time": query_time,
        "model_name": client.name,
        "embedding_dim": corpus_embs.shape[1],
    }


def main():
    parser = argparse.ArgumentParser(description="Uzbek Embedding Benchmark")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--local", action="store_true", help="Use sentence-transformers (local)")
    source.add_argument("--gemini", action="store_true", help="Use Gemini embedding API")

    parser.add_argument("--model-name", help="HuggingFace model name (e.g. BAAI/bge-m3)")
    parser.add_argument("--gemini-api-key", help="Gemini API key (or set GEMINI_API_KEY env)")
    parser.add_argument("--gemini-model", default="gemini-embedding-001", help="Gemini model name")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code for HF models")
    parser.add_argument("--device", default=None, help="Force device (cpu, mps, cuda)")

    parser.add_argument("--query-prefix", default="", help="Prefix for query texts (e.g. 'query: ')")
    parser.add_argument("--passage-prefix", default="", help="Prefix for passage texts (e.g. 'passage: ')")

    parser.add_argument("--query-prompt-name", default=None, help="ST prompt_name for queries")
    parser.add_argument("--passage-prompt-name", default=None, help="ST prompt_name for passages")
    parser.add_argument("--st-task", default=None, help="ST task kwarg (e.g. 'retrieval' for jina v5)")
    parser.add_argument("--transliterate", default=None, help="Transliteration mode (e.g. 'latin2cyrillic')")
    parser.add_argument("--max-seq-length", type=int, default=None, help="Cap tokenizer truncation length")

    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to dataset JSON")
    parser.add_argument("--output", help="Output JSON path (auto-generated if not set)")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--k-values", default="1,3,5,10", help="Comma-separated K values for metrics")

    args = parser.parse_args()
    k_values = [int(k) for k in args.k_values.split(",")]

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        console.print(f"[red]Dataset not found: {dataset_path}[/red]")
        sys.exit(1)
    dataset = load_dataset(str(dataset_path))

    console.print(f"[bold green]Uzbek Embedding Benchmark[/bold green]")
    console.print(f"  Corpus: {len(dataset['corpus'])} passages")
    console.print(f"  Queries: {len(dataset['queries'])} queries")

    if args.gemini:
        api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            console.print("[red]Gemini API key required (--gemini-api-key or GEMINI_API_KEY env)[/red]")
            sys.exit(1)
        result = run_gemini_benchmark(api_key, args.gemini_model, dataset, args.batch_size)
    else:
        if not args.model_name:
            console.print("[red]--model-name required when using --local[/red]")
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

    relevants = {q["id"]: q["relevant_ids"] for q in dataset["queries"]}
    metrics = compute_all_metrics(rankings, relevants, k_values=k_values)

    total_texts = len(dataset["corpus"]) + len(dataset["queries"])
    total_time = result["corpus_time"] + result["query_time"]
    metrics["avg_latency_ms"] = (total_time / total_texts) * 1000
    metrics["corpus_throughput"] = len(dataset["corpus"]) / result["corpus_time"]
    metrics["query_throughput"] = len(dataset["queries"]) / result["query_time"]

    console.print(f"\n[bold green]Results: {result['model_name']}[/bold green]")
    console.print(f"  Embedding dim: {result['embedding_dim']}")

    table = Table(title="Retrieval Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold white", justify="right")

    table.add_row("MRR", f"{metrics['mrr']:.4f}")
    for k in k_values:
        table.add_row(f"Hit Rate@{k}", f"{metrics[f'hit_rate@{k}']:.4f}")
        table.add_row(f"Recall@{k}", f"{metrics[f'recall@{k}']:.4f}")
        table.add_row(f"NDCG@{k}", f"{metrics[f'ndcg@{k}']:.4f}")

    table.add_section()
    table.add_row("Avg latency (ms/text)", f"{metrics['avg_latency_ms']:.2f}")
    table.add_row("Corpus throughput (texts/s)", f"{metrics['corpus_throughput']:.1f}")
    table.add_row("Query throughput (texts/s)", f"{metrics['query_throughput']:.1f}")

    console.print(table)

    output_data = {
        "model_name": result["model_name"],
        "embedding_dim": result["embedding_dim"],
        "dataset": str(dataset_path),
        "num_corpus": len(dataset["corpus"]),
        "num_queries": len(dataset["queries"]),
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
    }

    if args.output:
        output_path = args.output
    else:
        os.makedirs("results_news", exist_ok=True)
        safe_name = result["model_name"].replace("/", "_").replace(" ", "_")
        output_path = f"results_news/{safe_name}.json"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    console.print(f"\n[dim]Results saved to {output_path}[/dim]")


if __name__ == "__main__":
    main()
