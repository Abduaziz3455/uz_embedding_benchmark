# Uzbek Embedding Benchmark — Report

## TL;DR

For a self-hosted Uzbek RAG pipeline, **BGE-M3** is the best open model: it ties
`multilingual-e5-large` on hard-negative discrimination, has the widest
confidence margin of any open model, runs at ~20 ms/text on Apple Silicon,
and costs nothing. If you're willing to pay for API quality,
`gemini-embedding-001` is the ceiling.

### Top 5 (standard retrieval, by MRR)

| # | Model | MRR | HR@1 | NDCG@10 | Notes |
|---|---|---|---|---|---|
| 1 | `gemini-embedding-001` | **0.9104** | 0.8602 | 0.9292 | API, 3072d |
| 2 | `intfloat/multilingual-e5-large-instruct` | **0.8557** | 0.7858 | 0.8821 | 1024d, no prefix |
| 3 | `BAAI/bge-m3` | **0.8342** | 0.7595 | 0.8614 | 1024d, 8K ctx |
| 4 | `BAAI/bge-m3-unsupervised` | 0.8174 | 0.7353 | 0.8493 | 1024d |
| 5 | `intfloat/multilingual-e5-large` | 0.7812 | 0.6991 | 0.8146 | 1024d, `query: / passage:` |

## Dataset

- **7,078 passages, 2,017 queries**
- Source: [Dovud-Asadov/uzbek-embedding-dataset](https://huggingface.co/datasets/Dovud-Asadov/uzbek-embedding-dataset)
- Real passages from Uzbek news websites (kun.uz and others)
- Each query has 1 positive passage + 3 hard negatives (similar-but-wrong)
- Hardware for runs below: Apple M4 Pro, 24 GB unified memory, PyTorch MPS

## Part 1 — Standard Retrieval

Retrieval over the full 7,078-passage corpus. Higher is better for everything
except `ms/text` (latency).

| # | Model | Dim | MRR | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@10 | ms/text |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | gemini-embedding-001              | 3072 | **0.9104** | **0.8602** | **0.9529** | **0.9732** | **0.9881** | **0.9292** | 153.55 |
| 2  | multilingual-e5-large-instruct    | 1024 | 0.8557 | 0.7858 | 0.9118 | 0.9455 | 0.9673 | 0.8821 | 16.60 |
| 3  | bge-m3                            | 1024 | 0.8342 | 0.7595 | 0.8944 | 0.9217 | 0.9509 | 0.8614 | 18.05 |
| 4  | bge-m3-unsupervised               | 1024 | 0.8174 | 0.7353 | 0.8805 | 0.9222 | 0.9539 | 0.8493 | 17.62 |
| 5  | multilingual-e5-large             | 1024 | 0.7812 | 0.6991 | 0.8409 | 0.8795 | 0.9286 | 0.8146 | 18.07 |
| 6  | harrier-oss-v1-0.6b               | 1024 | 0.7228 | 0.6346 | 0.7843 | 0.8304 | 0.8870 | 0.7596 | 52.16 |
| 7  | multilingual-e5-base              |  768 | 0.7116 | 0.6182 | 0.7769 | 0.8260 | 0.8790 | 0.7486 |  5.84 |
| 8  | jina-embeddings-v5-text-small     | 1024 | 0.7025 | 0.6088 | 0.7690 | 0.8126 | 0.8691 | 0.7396 | 59.46 |
| 9  | jina-embeddings-v5-text-nano      |  768 | 0.6975 | 0.6063 | 0.7586 | 0.8081 | 0.8602 | 0.7330 | 15.04 |
| 10 | snowflake-arctic-embed-l-v2.0     | 1024 | 0.6723 | 0.5771 | 0.7298 | 0.7828 | 0.8577 | 0.7129 | 22.09 |
| 11 | Octen-Embedding-0.6B              | 1024 | 0.6042 | 0.5057 | 0.6619 | 0.7229 | 0.7878 | 0.6439 | 54.06 |
| 12 | multilingual-e5-small             |  384 | 0.6000 | 0.4879 | 0.6673 | 0.7318 | 0.8066 | 0.6444 | **2.61** |
| 13 | F2LLM-v2-0.6B                     | 1024 | 0.5517 | 0.4522 | 0.6034 | 0.6648 | 0.7372 | 0.5897 | 52.89 |
| 14 | Qwen3-Embedding-0.6B              | 1024 | 0.5272 | 0.4298 | 0.5821 | 0.6391 | 0.7075 | 0.5646 | 52.93 |
| 15 | embeddinggemma-300m               |  768 | 0.4289 | 0.3371 | 0.4700 | 0.5310 | 0.6034 | 0.4634 | 15.16 |
| 16 | paraphrase-multilingual           |  768 | 0.3829 | 0.2980 | 0.4135 | 0.4660 | 0.5553 | 0.4161 |  8.28 |
| 17 | tahrirchi-bert-base               |  768 | 0.0921 | 0.0501 | 0.0962 | 0.1190 | 0.1606 | 0.0995 |  5.10 |

### Metric definitions

- **MRR** — Mean Reciprocal Rank of the first relevant passage.
- **HR@K** — Hit Rate: fraction of queries with a relevant passage in top K.
- **R@K** — Recall at K.
- **NDCG@K** — Ranking quality with position discount.
- **ms/text** — Average embedding latency per text (lower is better).

## Part 2 — Hard-Negative Discrimination

Can the model tell apart topically-similar-but-wrong passages from the correct
one? This is the real failure mode in production RAG. Sorted by
**Discrimination Rate** — fraction of queries where the positive beats all 3
hard negatives.

| # | Model | Triplet Acc | **Discrim. Rate** | **Avg Margin** | Restricted MRR | Pos. Rank | HN Rank |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | gemini-embedding-001      | 0.9943 | **0.9861** | 0.1271 | 0.9930 | **1.6** | 106.8 |
| 2 | multilingual-e5-large     | 0.9825 | **0.9643** | 0.0480 | 0.9808 |  6.1 | 234.9 |
| 3 | bge-m3                    | 0.9825 | **0.9633** | **0.1829** | 0.9805 |  4.5 | 158.1 |
| 4 | snowflake-arctic-embed2   | 0.9349 | 0.8756 | 0.1383 | 0.9316 | 16.8 | 111.5 |
| 5 | Qwen3-Embedding-0.6B      | 0.9013 | 0.8215 | 0.1185 | 0.8999 | 95.3 | 314.5 |
| 6 | paraphrase-multilingual   | 0.8860 | 0.7992 | 0.1288 | 0.8863 | 140.6 | 502.1 |
| 7 | embeddinggemma-300m       | 0.8808 | 0.7794 | 0.1013 | 0.8773 | 142.0 | 348.0 |

### Metric definitions

- **Triplet Accuracy** — % of `(query, positive, negative)` pairs where the
  positive is ranked higher.
- **Discrimination Rate** — % of queries where the positive beats **all 3**
  hard negatives. The clutch metric for RAG.
- **Avg Margin** — Cosine similarity gap between positive and best negative;
  higher = more confident.
- **Restricted MRR** — MRR computed over only `{positive ∪ hard negatives}`.
- **Pos. Rank / HN Rank** — Average full-corpus rank of positive vs hardest
  negative (context for how far apart they land).

### Key findings

1. **Top 3 are tied on discrimination (≥96%)** — Gemini (98.6%),
   multilingual-e5-large (96.4%), bge-m3 (96.3%). The full-corpus MRR gap
   mostly reflects corpus-wide noise, not ability to pick the right passage
   among similar ones.
2. **BGE-M3 has the widest margin (0.183)** — nearly 4× the margin of
   multilingual-e5-large (0.048). In a RAG pipeline that uses score thresholds
   or weighted reranking, wider margins are more robust.
3. **Embedding space collapse** — Qwen3 / paraphrase-multilingual /
   embeddinggemma-300m can discriminate pairwise (78–82%) but their positive
   still sinks to rank 95–142 in the full corpus. Not recommended at scale.

## Recommendations

| Use case | Model | Why |
|----------|-------|-----|
| **Self-hosted RAG (default)** | **`BAAI/bge-m3`** | 96% discrimination, widest margin (0.18), ~20 ms/text, 8K ctx, free |
| **Maximum quality** | `gemini-embedding-001` | 99% discrimination, 0.91 MRR — but API cost + ~8× slower |
| **CPU-bound / smallest footprint** | `intfloat/multilingual-e5-small` | 0.60 MRR at ~2.6 ms/text (fastest), 384d |
| **Aggressive instruct baseline** | `intfloat/multilingual-e5-large-instruct` | Best open MRR (0.8557), no prefix needed |
| **Fallback** | `intfloat/multilingual-e5-large` | Ties bge-m3 on discrimination; thinner margin — less robust |

## Reproducing

```bash
# one model, standard retrieval
python run.py bench bge-m3

# one model, hard-negative analysis
python run.py bench bge-m3 --hard

# every configured model
python run.py bench-all

# compare everything
python run.py compare
python run.py compare --hard
```

Runs land in `results_news/<key>.json` (standard) or
`results_news/<key>_hard_neg.json` (hard-negative).
