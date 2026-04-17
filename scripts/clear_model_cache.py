#!/usr/bin/env python3
"""Delete HuggingFace-cached models listed in models_config.yaml.

Usage:
  python scripts/clear_model_cache.py             # dry run (shows what would be deleted)
  python scripts/clear_model_cache.py --yes       # actually delete
  python scripts/clear_model_cache.py --only bge-m3,multilingual-e5-large --yes
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "models_config.yaml"
HF_HUB_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def hf_name_to_cache_dir(hf_name: str) -> Path:
    return HF_HUB_CACHE / ("models--" + hf_name.replace("/", "--"))


def human_size(nbytes: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PiB"


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="Actually delete (otherwise dry run)")
    ap.add_argument("--only", help="Comma-separated model keys to target")
    ap.add_argument("--skip", help="Comma-separated model keys to skip")
    args = ap.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    keys = list(config["models"].keys())
    if args.only:
        wanted = {k.strip() for k in args.only.split(",")}
        keys = [k for k in keys if k in wanted]
    if args.skip:
        skip = {k.strip() for k in args.skip.split(",")}
        keys = [k for k in keys if k not in skip]

    total_bytes = 0
    targets: list[tuple[str, Path, int]] = []
    for key in keys:
        spec = config["models"][key]
        if spec.get("runtime") == "gemini":
            continue
        cache_dir = hf_name_to_cache_dir(spec["hf_name"])
        if cache_dir.exists():
            size = dir_size(cache_dir)
            targets.append((key, cache_dir, size))
            total_bytes += size

    if not targets:
        print(f"No cached models found under {HF_HUB_CACHE}")
        return

    print(f"Cache root: {HF_HUB_CACHE}")
    print(f"Found {len(targets)} cached model(s), total {human_size(total_bytes)}:\n")
    for key, path, size in targets:
        print(f"  [{human_size(size):>10}]  {key:35s}  {path}")

    if not args.yes:
        print("\nDry run. Re-run with --yes to actually delete.")
        return

    print()
    for key, path, _ in targets:
        print(f"Deleting {key} ({path}) ...")
        shutil.rmtree(path)
    print(f"\nFreed {human_size(total_bytes)}.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
