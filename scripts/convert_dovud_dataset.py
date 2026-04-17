#!/usr/bin/env python3
"""
Convert Dovud-Asadov/uzbek-embedding-dataset to our benchmark JSON format.

The dataset has query + positive + 3 hard negatives per row.
We build a deduplicated corpus from all passages and map queries to their
relevant passage IDs.

Usage:
  python scripts/convert_dovud_dataset.py --output dataset/uz_news_benchmark.json
"""

import argparse
import hashlib
import json
import sys

from datasets import load_dataset


def text_hash(text: str) -> str:
    """Short hash for deduplication."""
    return hashlib.md5(text.encode()).hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dataset/uz_news_benchmark.json")
    parser.add_argument("--split", default="test", help="Dataset split to use (test=2017, train=38321)")
    parser.add_argument("--max-queries", type=int, default=None, help="Limit number of queries")
    args = parser.parse_args()

    print("Loading dataset from HuggingFace...")
    ds = load_dataset("Dovud-Asadov/uzbek-embedding-dataset", split=args.split)
    print(f"  Loaded {len(ds)} rows from '{args.split}' split")

    # Build deduplicated corpus
    text_to_id = {}  # text -> passage_id
    corpus = []
    passage_counter = 0

    def get_or_create_passage(text: str) -> str:
        nonlocal passage_counter
        if not text or not text.strip():
            return None
        text = text.strip()
        h = text_hash(text)
        if h not in text_to_id:
            pid = f"p{passage_counter:05d}"
            text_to_id[h] = pid
            corpus.append({"id": pid, "text": text})
            passage_counter += 1
        return text_to_id[h]

    # Process rows
    queries = []
    rows = ds
    if args.max_queries:
        rows = ds.select(range(min(args.max_queries, len(ds))))

    skipped = 0
    for i, row in enumerate(rows):
        query_text = row["query"].strip() if row["query"] else ""
        if not query_text:
            skipped += 1
            continue

        # Register positive passage
        pos_id = get_or_create_passage(row["positive"])
        if not pos_id:
            skipped += 1
            continue

        # Register hard negatives
        hard_neg_ids = []
        for neg_key in ["negative_1", "negative_2", "negative_3"]:
            neg_text = row.get(neg_key, "")
            neg_id = get_or_create_passage(neg_text)
            if neg_id:
                hard_neg_ids.append(neg_id)

        queries.append({
            "id": f"q{i:05d}",
            "text": query_text,
            "relevant_ids": [pos_id],
            "hard_negative_ids": hard_neg_ids,
        })

    # Build output
    output = {
        "metadata": {
            "name": "Uzbek News Embedding Benchmark",
            "description": "Derived from Dovud-Asadov/uzbek-embedding-dataset (real Uzbek news articles)",
            "source": "https://huggingface.co/datasets/Dovud-Asadov/uzbek-embedding-dataset",
            "language": "uz",
            "split_used": args.split,
            "num_queries": len(queries),
            "num_corpus": len(corpus),
            "synthetic": False,
            "notes": "Passages from real Uzbek news (kun.uz etc). Queries may be LLM-generated from articles.",
        },
        "corpus": corpus,
        "queries": queries,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nDone!")
    print(f"  Queries: {len(queries)} (skipped {skipped})")
    print(f"  Corpus passages: {len(corpus)}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
