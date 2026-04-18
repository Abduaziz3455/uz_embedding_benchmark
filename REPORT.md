# Uzbek Embedding Benchmark — Report

## TL;DR

For a self-hosted Uzbek RAG pipeline, **`harrier-oss-v1-0.6b`** is the new top
open model: it wins both standard retrieval (MRR 0.8367) **and** hard-negative
discrimination (99.01% — the highest of any model, beating even Gemini).
**`BAAI/bge-m3`** remains the safest workhorse: near-tied on quality
(MRR 0.8342, 96.4% discrimination), 8K context, no prompt quirks, and a
long track record. If you're willing to pay for
API quality, `gemini-embedding-001` still sets the ceiling on raw MRR but is
slightly behind `harrier-oss-v1-0.6b` on discrimination.

### Top 5 (standard retrieval, by MRR)

| # | Model | MRR | HR@1 | NDCG@10 | Notes |
|---|---|---|---|---|---|
| 1 | `gemini-embedding-001` | **0.9104** | 0.8602 | 0.9292 | API, 3072d |
| 2 | `microsoft/harrier-oss-v1-0.6b` | **0.8367** | 0.7600 | 0.8659 | 1024d, needs `web_search_query` prompt |
| 3 | `BAAI/bge-m3` | **0.8342** | 0.7595 | 0.8614 | 1024d, 8K ctx |
| 4 | `nomic-ai/nomic-embed-text-v2-moe` | 0.8181 | 0.7432 | 0.8487 | 768d, MoE, `search_query:` prefix |
| 5 | `BAAI/bge-m3-unsupervised` | 0.8174 | 0.7353 | 0.8493 | 1024d |

## Dataset

- **7,078 passages, 2,017 queries**
- Source: [Dovud-Asadov/uzbek-embedding-dataset](https://huggingface.co/datasets/Dovud-Asadov/uzbek-embedding-dataset)
- Real passages from Uzbek news websites (kun.uz and others)
- Each query has 1 positive passage + 3 hard negatives (similar-but-wrong)

## Hardware

- **GPU:** NVIDIA RTX 5070 (Blackwell, 12 GB VRAM)
- **Runtime:** PyTorch CUDA via `Dockerfile.gpu` (pytorch/pytorch CUDA 12.8 runtime)
- 25 models ran end-to-end; no OOM or crash failures in this run

Latencies below (`ms/text`) are GPU-side; they will be ~5–20× higher on CPU
or Apple Silicon (MPS). Throughput cells in Part 1 come directly from the
per-run `corpus_throughput` and `query_throughput` fields.

> **Hardware disclaimer:** all three top open models are ≤0.6B
> parameters, so you can comfortably run any of them on a 16 GB
> M-series MacBook or any GPU with ≥6 GB VRAM (T4, RTX 3050, even
> Colab's free tier). No A100 required.

## Part 1 — Standard Retrieval

Retrieval over the full 7,078-passage corpus. Higher is better for
everything except `ms/text` (latency). All 25 models shown, sorted by MRR.

| # | Model | Dim | MRR | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@10 | ms/text |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | gemini-embedding-001                          | 3072 | **0.9104** | **0.8602** | **0.9529** | **0.9732** | **0.9881** | **0.9292** | 67.91 |
| 2  | harrier-oss-v1-0.6b                           | 1024 | 0.8367 | 0.7600 | 0.8999 | 0.9336 | 0.9603 | 0.8659 |  5.07 |
| 3  | bge-m3                                        | 1024 | 0.8342 | 0.7595 | 0.8944 | 0.9217 | 0.9509 | 0.8614 |  5.14 |
| 4  | nomic-embed-text-v2-moe                       |  768 | 0.8181 | 0.7432 | 0.8741 | 0.9127 | 0.9504 | 0.8487 |  2.55 |
| 5  | bge-m3-unsupervised                           | 1024 | 0.8174 | 0.7353 | 0.8805 | 0.9222 | 0.9539 | 0.8493 |  5.14 |
| 6  | multilingual-e5-large                         | 1024 | 0.7812 | 0.6991 | 0.8409 | 0.8795 | 0.9286 | 0.8146 |  5.22 |
| 7  | multilingual-e5-large-instruct                | 1024 | 0.7327 | 0.6366 | 0.7972 | 0.8547 | 0.9063 | 0.7720 |  1.45 |
| 8  | multilingual-e5-base                          |  768 | 0.7116 | 0.6182 | 0.7769 | 0.8260 | 0.8790 | 0.7486 |  1.63 |
| 9  | jina-embeddings-v5-text-small                 | 1024 | 0.7022 | 0.6073 | 0.7700 | 0.8126 | 0.8691 | 0.7394 |  8.62 |
| 10 | jina-embeddings-v5-text-nano                  |  768 | 0.6975 | 0.6063 | 0.7586 | 0.8081 | 0.8602 | 0.7330 |  4.02 |
| 11 | KaLM-embedding-mini-instruct-v2.5             |  896 | 0.6815 | 0.5875 | 0.7382 | 0.7908 | 0.8577 | 0.7204 |  9.42 |
| 12 | snowflake-arctic-embed-l-v2.0                 | 1024 | 0.6720 | 0.5771 | 0.7298 | 0.7838 | 0.8582 | 0.7128 |  5.21 |
| 13 | Octen-Embedding-0.6B                          | 1024 | 0.6032 | 0.5027 | 0.6619 | 0.7263 | 0.7863 | 0.6428 |  5.20 |
| 14 | granite-embedding-278m-multilingual           |  768 | 0.6005 | 0.5047 | 0.6569 | 0.7134 | 0.7749 | 0.6372 |  0.56 |
| 15 | multilingual-e5-small                         |  384 | 0.6000 | 0.4879 | 0.6673 | 0.7318 | 0.8066 | 0.6444 |  0.61 |
| 16 | bge-m3-uzbek-cyrillic                         | 1024 | 0.5966 | 0.4854 | 0.6634 | 0.7248 | 0.8047 | 0.6411 |  7.30 |
| 17 | KaLM-embedding-mini-v1                        |  896 | 0.5946 | 0.4943 | 0.6549 | 0.7100 | 0.7824 | 0.6350 |  8.96 |
| 18 | F2LLM-v2-0.6B                                 | 1024 | 0.5499 | 0.4492 | 0.6029 | 0.6624 | 0.7397 | 0.5891 |  5.06 |
| 19 | Qwen3-Embedding-0.6B                          | 1024 | 0.5276 | 0.4323 | 0.5796 | 0.6391 | 0.7080 | 0.5650 |  5.10 |
| 20 | granite-embedding-107m-multilingual           |  384 | 0.4620 | 0.3669 | 0.5047 | 0.5706 | 0.6435 | 0.4981 | **0.26** |
| 21 | embeddinggemma-300m                           |  768 | 0.4289 | 0.3371 | 0.4700 | 0.5310 | 0.6034 | 0.4634 |  3.41 |
| 22 | ModernUzBERT                                  |  768 | 0.3905 | 0.2856 | 0.4358 | 0.5032 | 0.6014 | 0.4321 |  1.87 |
| 23 | paraphrase-multilingual-mpnet-base-v2         |  768 | 0.3827 | 0.2980 | 0.4130 | 0.4670 | 0.5538 | 0.4156 |  1.06 |
| 24 | uz_embeddinggemma-300m                        |  768 | 0.3715 | 0.2811 | 0.4065 | 0.4596 | 0.5518 | 0.4063 |  3.42 |
| 25 | tahrirchi-bert-base                           |  768 | 0.0921 | 0.0501 | 0.0962 | 0.1190 | 0.1606 | 0.0995 |  1.45 |

### Metric definitions

- **MRR** — Mean Reciprocal Rank of the first relevant passage.
- **HR@K** — Hit Rate: fraction of queries with the relevant passage in top K.
- **NDCG@K** — Ranking quality with position discount.
- **ms/text** — Average embedding latency per text on the RTX 5070
  (lower is better; pure forward-pass time, not including I/O).

## Part 2 — Hard-Negative Discrimination

Can the model tell the correct passage apart from topically-similar-but-wrong
ones? This is the real failure mode in production RAG. All 25 models, sorted
by **Discrimination Rate** (fraction of queries where the positive beats
all 3 hard negatives).

| # | Model | Triplet Acc | **Discrim. Rate** | **Avg Margin** | Restricted MRR | Pos. Rank | HN Rank |
|---|---|---:|---:|---:|---:|---:|---:|
| 1  | harrier-oss-v1-0.6b                           | 0.9949 | **0.9901** | 0.1691 | 0.9946 | **3.2** | 436.6 |
| 2  | gemini-embedding-001                          | 0.9943 | 0.9861 | 0.1271 | 0.9930 | **1.6** | 106.8 |
| 3  | multilingual-e5-large                         | 0.9825 | 0.9643 | 0.0480 | 0.9808 |  6.1 | 234.9 |
| 4  | bge-m3                                        | 0.9827 | 0.9638 | 0.1827 | 0.9808 |  4.5 | 157.8 |
| 5  | nomic-embed-text-v2-moe                       | 0.9815 | 0.9598 | **0.2175** | 0.9789 |  3.6 | 201.0 |
| 6  | bge-m3-unsupervised                           | 0.9801 | 0.9584 | 0.1568 | 0.9778 |  3.4 | 148.3 |
| 7  | multilingual-e5-base                          | 0.9760 | 0.9509 | 0.0457 | 0.9736 | 12.8 | 386.1 |
| 8  | multilingual-e5-small                         | 0.9738 | 0.9445 | 0.0424 | 0.9705 | 18.6 | 410.6 |
| 9  | multilingual-e5-large-instruct                | 0.9721 | 0.9415 | 0.0382 | 0.9689 |  9.0 | 221.3 |
| 10 | bge-m3-uzbek-cyrillic                         | 0.9666 | 0.9321 | 0.1952 | 0.9634 | 13.9 | 439.8 |
| 11 | KaLM-embedding-mini-instruct-v2.5             | 0.9652 | 0.9251 | 0.1285 | 0.9605 | 25.5 | 303.7 |
| 12 | jina-embeddings-v5-text-small                 | 0.9575 | 0.9157 | 0.2042 | 0.9543 | 14.6 | 183.5 |
| 13 | jina-embeddings-v5-text-nano                  | 0.9579 | 0.9137 | 0.1926 | 0.9538 | 13.2 | 179.9 |
| 14 | KaLM-embedding-mini-v1                        | 0.9418 | 0.8914 | 0.1293 | 0.9400 | 56.1 | 399.5 |
| 15 | granite-embedding-278m-multilingual           | 0.9424 | 0.8850 | 0.0807 | 0.9377 | 36.9 | 288.5 |
| 16 | Octen-Embedding-0.6B                          | 0.9420 | 0.8845 | 0.1633 | 0.9374 | 46.2 | 356.5 |
| 17 | snowflake-arctic-embed-l-v2.0                 | 0.9343 | 0.8751 | 0.1381 | 0.9313 | 16.8 | 111.4 |
| 18 | F2LLM-v2-0.6B                                 | 0.9225 | 0.8468 | 0.1590 | 0.9168 | 46.0 | 279.1 |
| 19 | granite-embedding-107m-multilingual           | 0.9123 | 0.8285 | 0.0588 | 0.9065 | 97.4 | 449.5 |
| 20 | Qwen3-Embedding-0.6B                          | 0.9007 | 0.8190 | 0.1185 | 0.8988 | 95.2 | 314.7 |
| 21 | paraphrase-multilingual-mpnet-base-v2         | 0.8858 | 0.7987 | 0.1289 | 0.8861 | 140.7 | 502.9 |
| 22 | embeddinggemma-300m                           | 0.8808 | 0.7794 | 0.1013 | 0.8773 | 142.0 | 348.0 |
| 23 | uz_embeddinggemma-300m                        | 0.8777 | 0.7700 | 0.1287 | 0.8728 | 148.7 | 408.7 |
| 24 | ModernUzBERT                                  | 0.8042 | 0.6906 | 0.0549 | 0.8177 | 76.8 | 125.6 |
| 25 | tahrirchi-bert-base                           | 0.6180 | 0.4303 | -0.0099 | 0.6570 | 606.4 | 309.5 |

### Metric definitions

- **Triplet Accuracy** — % of `(query, positive, negative)` pairs where the
  positive is ranked higher than the negative.
- **Discrimination Rate** — % of queries where the positive beats **all 3**
  hard negatives. The clutch metric for RAG.
- **Avg Margin** — Cosine-similarity gap between positive and best negative;
  higher = more confident separation.
- **Restricted MRR** — MRR computed over only `{positive ∪ 3 hard negatives}`.
- **Pos. Rank / HN Rank** — Average full-corpus rank of positive vs hardest
  negative (context for how far apart they land in the 7,078-passage corpus).

## Key findings

1. **`harrier-oss-v1-0.6b` is the new open-model leader.** Once the model is
   given its expected `web_search_query` prompt name, it becomes the top open
   model on MRR (0.8367) *and* the top model overall on discrimination
   (99.01%) — even ahead of Gemini (98.61%). It also has the smallest average
   positive-rank of any open model (3.2).
2. **`bge-m3` is still the stable default.** MRR within 0.003 of harrier,
   96.4% discrimination, 8K context, no prompt quirks, and a long track
   record. If you want one model that will not surprise you, pick this.
3. **`nomic-embed-text-v2-moe` is the efficiency pick.** 768-dim MoE model
   that lands in the top 5 on MRR, has the widest margin of *any* model
   (0.2175), and runs at 2.55 ms/text on the 5070 — the fastest non-tiny
   model in the bench. Requires the `search_query: / search_document:`
   prefixes.
4. **`multilingual-e5-large-instruct` dropped vs. older runs.** In the
   earlier Apple-Silicon runs it led the open board (MRR ~0.856); in this
   CUDA run it lands at 7th (0.7327). The likely cause is that the instruct
   variant really wants a task-specific instruct string in front of the
   query, which we are not passing. Treat the number as a baseline
   for "no prompt engineering."
5. **Gemini still wins raw retrieval.** 0.9104 MRR is ~9 points ahead of the
   best open model. If budget allows and API latency is acceptable, it is
   still the quality ceiling — but harrier now matches it on the
   RAG-relevant discrimination metric.
6. **Bottom half is embedding-space collapse.** `Qwen3-Embedding-0.6B`,
   `paraphrase-multilingual`, `embeddinggemma-300m`, `uz_embeddinggemma-300m`,
   and `ModernUzBERT` can do pairwise discrimination (≥69%) but their
   positive sinks to rank 77–148 in the 7,078-passage corpus. Not usable
   at scale without re-ranking.
7. **`tahrirchi-bert-base` is a masked-LM, not an embedding model.** MRR
   0.09 and a **negative** average margin means the pooled CLS vector is
   essentially random for retrieval. Included as a sanity floor, not a
   recommendation.

## Recommendations

| Use case | Model | Why |
|----------|-------|-----|
| **Self-hosted RAG (new default)** | **`microsoft/harrier-oss-v1-0.6b`** | Highest open-model MRR (0.8367), highest discrimination of any model (99%), 5 ms/text on RTX 5070 |
| **Self-hosted RAG (safe default)** | **`BAAI/bge-m3`** | Tied on quality (MRR 0.8342, 96.4% disc.), 8K context, no prompt quirks, long track record |
| **Efficiency / 768-dim index** | `nomic-ai/nomic-embed-text-v2-moe` | MRR 0.8181, widest margin (0.22), 2.5 ms/text, MoE |
| **Maximum quality** | `gemini-embedding-001` | 0.9104 MRR — but API cost and ~13× slower than top open models |
| **Smallest footprint** | `intfloat/multilingual-e5-small` | 0.60 MRR at 0.6 ms/text, 384d — good for constrained infra |
| **Cyrillic-only Uzbek corpus** | `Just-Bax/bge-m3-uzbek-cyrillic` | Purpose-built Latin→Cyrillic transliteration path, MRR 0.60 |
| **Avoid at this corpus size** | Qwen3-0.6B, `embeddinggemma-300m`, `paraphrase-multilingual`, `tahrirchi-bert-base` | Rank collapse in full-corpus retrieval |

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
