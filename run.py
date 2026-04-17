#!/usr/bin/env python3
"""
Uzbek Embedding Benchmark — unified runner.

  python run.py list                          # list models in models_config.yaml
  python run.py bench bge-m3                  # standard retrieval
  python run.py bench bge-m3 --hard           # hard-negative analysis
  python run.py bench-all                     # run standard for every model
  python run.py bench-all --hard              # run hard-neg for every model
  python run.py bench-all --only bge-m3,multilingual-e5-large
  python run.py compare                       # compare all standard results
  python run.py compare --hard                # compare all hard-neg results

Model definitions live in `models_config.yaml`. Edit it to add or tune models.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Missing dependency: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "models_config.yaml"
RESULTS_DIR = ROOT / "results_news"
DEFAULT_DATASET = ROOT / "dataset" / "uz_news_benchmark.json"

console = Console()


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        console.print(f"[red]Config not found: {CONFIG_PATH}[/red]")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_cmd(key: str, spec: dict, hard: bool, dataset: str, batch_override: int | None) -> list[str]:
    script = str(ROOT / ("scripts/hard_negative_analysis.py" if hard else "benchmark.py"))
    out_name = f"{key}_hard_neg.json" if hard else f"{key}.json"
    output = RESULTS_DIR / out_name

    runtime = spec.get("runtime", "local")
    cmd = [sys.executable, script, "--dataset", dataset, "--output", str(output)]

    if runtime == "gemini":
        cmd += ["--gemini", "--gemini-model", spec.get("hf_name", "gemini-embedding-001")]
    else:  # local
        cmd += ["--local", "--model-name", spec["hf_name"]]

    if spec.get("trust_remote_code"):
        cmd.append("--trust-remote-code")
    for flag, field in [
        ("--query-prefix", "query_prefix"),
        ("--passage-prefix", "passage_prefix"),
        ("--query-prompt-name", "query_prompt_name"),
        ("--passage-prompt-name", "passage_prompt_name"),
        ("--st-task", "st_task"),
        ("--transliterate", "transliterate"),
    ]:
        val = spec.get(field)
        if val is not None and val != "":
            cmd += [flag, str(val)]

    bs = batch_override or spec.get("batch_size")
    if bs:
        cmd += ["--batch-size", str(bs)]

    return cmd


def cmd_list(args):
    config = load_config()
    table = Table(title=f"Configured models ({len(config['models'])})")
    table.add_column("Key", style="cyan")
    table.add_column("Runtime", style="yellow")
    table.add_column("HF / API model", style="white")
    table.add_column("Notes", style="dim")
    for key, spec in config["models"].items():
        runtime = spec.get("runtime", "local")
        notes = []
        if spec.get("trust_remote_code"):
            notes.append("trust_remote_code")
        if spec.get("query_prefix"):
            notes.append(f"prefix='{spec['query_prefix'].strip()}'")
        if spec.get("query_prompt_name"):
            notes.append(f"prompt={spec['query_prompt_name']}")
        if spec.get("st_task"):
            notes.append(f"task={spec['st_task']}")
        if spec.get("transliterate"):
            notes.append(f"translit={spec['transliterate']}")
        if spec.get("batch_size"):
            notes.append(f"bs={spec['batch_size']}")
        table.add_row(key, runtime, spec.get("hf_name", "—"), ", ".join(notes))
    console.print(table)


def cmd_bench(args):
    config = load_config()
    if args.model not in config["models"]:
        console.print(f"[red]Unknown model key: {args.model}[/red]")
        console.print("Available: " + ", ".join(config["models"].keys()))
        sys.exit(1)
    spec = config["models"][args.model]
    out_name = f"{args.model}_hard_neg.json" if args.hard else f"{args.model}.json"
    output_path = RESULTS_DIR / out_name
    if output_path.exists() and not args.force:
        console.print(f"[yellow]↷ {args.model}: result already exists at {output_path}, skipping (use --force to re-run)[/yellow]")
        return
    cmd = build_cmd(args.model, spec, hard=args.hard, dataset=args.dataset, batch_override=args.batch_size)
    console.print(f"[bold cyan]→ {' '.join(cmd)}[/bold cyan]")
    raise SystemExit(subprocess.call(cmd))


def cmd_bench_all(args):
    config = load_config()
    keys = list(config["models"].keys())
    if args.only:
        wanted = {k.strip() for k in args.only.split(",")}
        keys = [k for k in keys if k in wanted]
        missing = wanted - set(keys)
        if missing:
            console.print(f"[yellow]Skipping unknown keys:[/yellow] {', '.join(missing)}")
    if args.skip:
        skip = {k.strip() for k in args.skip.split(",")}
        keys = [k for k in keys if k not in skip]

    console.print(f"[bold green]Running {len(keys)} model(s){' (hard-negative)' if args.hard else ''}[/bold green]")
    failures = []
    skipped = []
    for i, key in enumerate(keys, 1):
        spec = config["models"][key]
        console.rule(f"[{i}/{len(keys)}] {key}")
        out_name = f"{key}_hard_neg.json" if args.hard else f"{key}.json"
        output_path = RESULTS_DIR / out_name
        if output_path.exists() and not args.force:
            console.print(f"[yellow]↷ {key}: result already exists at {output_path}, skipping (use --force to re-run)[/yellow]")
            skipped.append(key)
            continue
        cmd = build_cmd(key, spec, hard=args.hard, dataset=args.dataset, batch_override=args.batch_size)
        rc = subprocess.call(cmd)
        if rc != 0:
            failures.append(key)
            console.print(f"[red]✗ {key} failed (exit {rc})[/red]")
            if not args.keep_going:
                sys.exit(rc)
    console.rule("Done")
    if skipped:
        console.print(f"[yellow]Skipped (already done): {', '.join(skipped)}[/yellow]")
    if failures:
        console.print(f"[red]Failed: {', '.join(failures)}[/red]")
        sys.exit(1)


def cmd_compare(args):
    paths = sorted(RESULTS_DIR.glob("*_hard_neg.json" if args.hard else "*.json"))
    if not args.hard:
        paths = [p for p in paths if not p.name.endswith("_hard_neg.json")]
    if not paths:
        console.print(f"[red]No result files in {RESULTS_DIR}[/red]")
        sys.exit(1)

    results = [json.load(open(p)) for p in paths]

    if args.hard:
        console.print(f"\n[bold green]Hard-Negative Comparison ({len(results)} models)[/bold green]\n")
        rows = [(r["model_name"], r["hard_negative_metrics"]) for r in results]
        rows.sort(key=lambda r: -r[1]["discrimination_rate"])

        table = Table(title="Hard-Negative Metrics (sorted by Discrimination Rate)", show_lines=False)
        table.add_column("#", style="dim")
        table.add_column("Model", style="cyan")
        table.add_column("Discrim.", justify="right")
        table.add_column("Triplet", justify="right")
        table.add_column("Margin", justify="right")
        table.add_column("R-MRR", justify="right")
        table.add_column("Pos rk", justify="right")
        table.add_column("HN rk", justify="right")
        for i, (name, hn) in enumerate(rows, 1):
            table.add_row(
                str(i), name,
                f"{hn['discrimination_rate']:.4f}",
                f"{hn['triplet_accuracy']:.4f}",
                f"{hn['avg_margin']:.4f}",
                f"{hn['restricted_mrr']:.4f}",
                f"{hn['positive_avg_rank']:.1f}",
                f"{hn['hard_neg_avg_rank']:.1f}",
            )
        console.print(table)
    else:
        console.print(f"\n[bold green]Standard Retrieval Comparison ({len(results)} models)[/bold green]\n")
        rows = [(r["model_name"], r["embedding_dim"], r["metrics"]) for r in results]
        rows.sort(key=lambda r: -r[2]["mrr"])

        table = Table(title="Standard Metrics (sorted by MRR)", show_lines=False)
        table.add_column("#", style="dim")
        table.add_column("Model", style="cyan")
        table.add_column("Dim", justify="right")
        table.add_column("MRR", justify="right")
        table.add_column("HR@1", justify="right")
        table.add_column("HR@5", justify="right")
        table.add_column("R@10", justify="right")
        table.add_column("NDCG@10", justify="right")
        table.add_column("ms/txt", justify="right")
        for i, (name, dim, m) in enumerate(rows, 1):
            table.add_row(
                str(i), name, str(dim),
                f"{m['mrr']:.4f}",
                f"{m['hit_rate@1']:.4f}",
                f"{m['hit_rate@5']:.4f}",
                f"{m['recall@10']:.4f}",
                f"{m['ndcg@10']:.4f}",
                f"{m.get('avg_latency_ms', 0):.1f}",
            )
        console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Uzbek Embedding Benchmark runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List configured models")

    p_bench = sub.add_parser("bench", help="Benchmark one model")
    p_bench.add_argument("model", help="Model key from models_config.yaml")
    p_bench.add_argument("--hard", action="store_true", help="Run hard-negative analysis instead of standard metrics")
    p_bench.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p_bench.add_argument("--batch-size", type=int, default=None)
    p_bench.add_argument("--force", action="store_true", help="Re-run even if a result file already exists")

    p_all = sub.add_parser("bench-all", help="Benchmark every configured model")
    p_all.add_argument("--hard", action="store_true")
    p_all.add_argument("--only", help="Comma-separated subset of model keys to run")
    p_all.add_argument("--skip", help="Comma-separated model keys to skip")
    p_all.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p_all.add_argument("--batch-size", type=int, default=None)
    p_all.add_argument("--keep-going", action="store_true", help="Continue even if a model fails")
    p_all.add_argument("--force", action="store_true", help="Re-run even if a result file already exists")

    p_cmp = sub.add_parser("compare", help="Compare all saved results")
    p_cmp.add_argument("--hard", action="store_true", help="Compare hard-negative results")

    args = parser.parse_args()
    {"list": cmd_list, "bench": cmd_bench, "bench-all": cmd_bench_all, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
