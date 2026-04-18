# Uzbek Embedding Benchmark

Benchmarks embedding models for Uzbek retrieval-augmented generation (RAG) on
a dataset of **7,078 passages and 2,017 queries** drawn from real Uzbek news
articles. Each query comes with one positive passage and three hard negatives.

**Latest run: 25 models on NVIDIA RTX 5070 (CUDA).** See
[REPORT.md](REPORT.md) for the full results and recommendations.

### Headline numbers (top 5 by MRR)

| # | Model | MRR | Discrim. Rate | ms/text |
|---|---|---:|---:|---:|
| 1 | `gemini-embedding-001` (API) | **0.9104** | 0.9861 | 67.9 |
| 2 | `microsoft/harrier-oss-v1-0.6b` | **0.8367** | **0.9901** |  5.1 |
| 3 | `BAAI/bge-m3` | **0.8342** | 0.9638 |  5.1 |
| 4 | `nomic-ai/nomic-embed-text-v2-moe` | 0.8181 | 0.9598 |  2.5 |
| 5 | `BAAI/bge-m3-unsupervised` | 0.8174 | 0.9584 |  5.1 |

## TL;DR — two commands

```bash
./bench bench bge-m3             # benchmark one model (auto CPU/GPU, builds on first run)
./bench compare                  # show ranked table of all results
```

That's it. Add `--hard` to any `bench*` / `compare` command to use
hard-negative analysis instead of standard retrieval metrics.

## What you can do

- Run any configured model with one command.
- Run all configured models sequentially.
- Score models on standard retrieval metrics (MRR, Recall@K, NDCG) **and**
  hard-negative discrimination (the real RAG failure mode).
- Compare saved results in a ranked table.
- Bring your own HuggingFace model by editing `models_config.yaml`.

Supported runtimes:

- **Local (HuggingFace / sentence-transformers)** — default. Works on CPU, Apple Silicon (MPS), or CUDA.
- **Gemini API** — for `gemini-embedding-001`.

## Quick start

### Option A — Docker, auto-GPU (recommended)

The `./bench` wrapper:

1. Checks whether the host has an NVIDIA GPU (`nvidia-smi`).
2. Picks the right image — a **slim CPU Python image** (~1 GB) on CPU-only hosts,
   or the **PyTorch CUDA runtime** (~6 GB) on GPU hosts.
3. Builds the image the first time it's needed, then reuses it.
4. Runs the benchmark inside the container.

You don't pick; it does. On any machine, everything after `./bench …` is identical.

```bash
./bench list                     # what's configured
./bench bench bge-m3             # benchmark one model
./bench bench bge-m3 --hard      # hard-negative analysis
./bench bench-all                # every configured model sequentially
./bench compare                  # ranked comparison of saved results
./bench compare --hard           # same, hard-negative table
```

Results land in `./results_news/` on the host. The HuggingFace cache is
mounted from `~/.cache/huggingface` so downloaded weights are reused across
runs and across any other project sharing the cache.

GPU requirements (Linux only):
- NVIDIA driver (tested against RTX 5070 / CUDA 12.8)
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  so Docker can see the GPU

Gemini model in Docker:

```bash
cp .env.example .env
# edit .env and set GEMINI_API_KEY
./bench bench gemini-embedding-001
```

### Option B — Local Python

```bash
# 1. Create a virtualenv (Python 3.10+ required)
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install torch           # or: pip install torch --index-url https://download.pytorch.org/whl/cu121  (CUDA)
pip install -r requirements.txt

# 3. List models
python run.py list

# 4. Benchmark
python run.py bench bge-m3
python run.py bench bge-m3 --hard
python run.py bench-all
python run.py compare
python run.py compare --hard
```

For the Gemini model, set the API key first:

```bash
cp .env.example .env
# edit .env and set GEMINI_API_KEY
export $(cat .env | xargs)
python run.py bench gemini-embedding-001
```

## Common workflows

```bash
# Only benchmark a subset
python run.py bench-all --only bge-m3,harrier-oss-v1-0.6b,nomic-embed-text-v2-moe

# Skip specific models
python run.py bench-all --skip tahrirchi-bert-base

# Continue past failures (useful for overnight runs)
python run.py bench-all --keep-going

# Override batch size (helpful on low-RAM machines)
python run.py bench bge-m3 --batch-size 8
```

## Adding a new model

Edit [`models_config.yaml`](models_config.yaml) and append an entry. Keys
become the CLI names.

```yaml
models:
  my-new-model:
    hf_name: org/repo-name
    trust_remote_code: true          # if the HF model requires it
    query_prefix: "query: "          # optional — e5-style prefix
    passage_prefix: "passage: "
    query_prompt_name: query         # optional — ST prompt_name
    passage_prompt_name: document
    st_task: retrieval               # optional — ST task kwarg
    batch_size: 16                   # optional — override default 32
    transliterate: latin2cyrillic    # optional — Cyrillic-only models
```

Then:

```bash
python run.py bench my-new-model
```

## Hardware used for the numbers in REPORT.md

- **GPU:** NVIDIA RTX 5070 (Blackwell, 12 GB VRAM)
- **Container runtime:** `Dockerfile.gpu` (pytorch/pytorch CUDA 12.8 runtime)
- 25 models ran end-to-end; no OOM or crash failures

Apple Silicon (MPS) and CPU paths work the same — only the per-model
`ms/text` and throughput change. Retrieval metrics are
hardware-independent.

> **Hardware disclaimer:** all three top open models are ≤0.6B
> parameters, so you can comfortably run any of them on a 16 GB
> M-series MacBook or any GPU with ≥6 GB VRAM (T4, RTX 3050, even
> Colab's free tier). No A100 required.

## Known issues

### KaLM-embedding-multilingual-mini-instruct-v2.5 needs transformers v4

This model ships a custom `modeling.py` written against `transformers==4.45`.
On `transformers>=5.0` it still loads (no crash), but the v4-era attention-mask
internals it imports behave differently under v5, silently producing
near-random embeddings (MRR ~0.01 instead of ~0.68).

The rest of the benchmark is happy on v5, so we don't pin globally. Run this
one model in a throwaway container with a v4 downgrade:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm \
  --entrypoint bash benchmark -c "\
    pip install 'transformers>=4.45,<5' 'sentence-transformers<4' && \
    python run.py bench kalm-embedding-mini-instruct-v2.5 --force"
```

`--rm` tears down the container after the run, so your main image keeps v5 and
other models are unaffected.

## Regenerating the dataset

The dataset is already checked in at `dataset/uz_news_benchmark.json`. If you
want to regenerate it from the HuggingFace source:

```bash
python scripts/convert_dovud_dataset.py --split test
```

## Project layout

```
bench                    # ./bench <subcommand>  — auto-GPU Docker wrapper
run.py                   # python run.py <subcommand>  — bare-metal runner
benchmark.py             # standard retrieval engine
scripts/hard_negative_analysis.py  # hard-negative analysis engine
models.py                # embedding clients (sentence-transformers, Gemini)
metrics.py               # MRR, Recall@K, NDCG@K
models_config.yaml       # model registry (edit to add models)
dataset/                 # benchmark dataset JSON
results_news/            # per-model result JSONs
scripts/convert_dovud_dataset.py   # rebuild dataset from HF
Dockerfile               # slim CPU image (python:3.12-slim + cpu torch)
Dockerfile.gpu           # CUDA image (pytorch/pytorch:…-cuda12.8-cudnn9-runtime)
docker-compose.yml       # CPU default
docker-compose.gpu.yml   # override: switches to Dockerfile.gpu + NVIDIA reservation
REPORT.md                # benchmark results + analysis
```
