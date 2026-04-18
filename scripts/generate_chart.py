"""Generate a 1200x1200 summary chart from results_news/ JSONs.

Outputs: benchmark_chart.png — two horizontal bar panels (MRR + Discrimination Rate)
for the top 10 models by each metric, with the three recommended picks and
Gemini highlighted.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results_news"
OUT = ROOT / "benchmark_chart.png"

GEMINI = "gemini-embedding-001"
NUM_PICKS = 3

DISPLAY_NAMES = {
    "gemini-embedding-001": "Gemini (API)",
    "harrier-oss-v1-0.6b": "harrier-oss-v1-0.6b",
    "bge-m3": "BAAI/bge-m3",
    "nomic-embed-text-v2-moe": "nomic-embed-text-v2-moe",
    "bge-m3-unsupervised": "bge-m3-unsupervised",
    "multilingual-e5-large": "multilingual-e5-large",
    "multilingual-e5-large-instruct": "multilingual-e5-large-instruct",
    "multilingual-e5-base": "multilingual-e5-base",
    "jina-embeddings-v5-text-small": "jina-v5-small",
    "jina-embeddings-v5-text-nano": "jina-v5-nano",
}

COLOR_PICK = "#10b981"
COLOR_GEMINI = "#8b5cf6"
COLOR_OTHER = "#cbd5e1"
COLOR_TEXT = "#0f172a"
COLOR_MUTED = "#64748b"


def load_rows():
    rows = []
    for std_path in RESULTS.glob("*.json"):
        if std_path.stem.endswith("_hard_neg"):
            continue
        hard_path = std_path.with_name(f"{std_path.stem}_hard_neg.json")
        if not hard_path.exists():
            continue
        std = json.loads(std_path.read_text())
        hard = json.loads(hard_path.read_text())
        key = std_path.stem
        rows.append(
            {
                "key": key,
                "mrr": std["metrics"]["mrr"],
                "disc": hard["hard_negative_metrics"]["discrimination_rate"],
            }
        )
    return rows


def top_by(rows, key, n=10):
    return sorted(rows, key=lambda r: r[key], reverse=True)[:n]


def color_for(key, picks):
    if key == GEMINI:
        return COLOR_GEMINI
    if key in picks:
        return COLOR_PICK
    return COLOR_OTHER


def draw_panel(ax, rows, value_key, title, xlim, picks):
    labels = [DISPLAY_NAMES.get(r["key"], r["key"]) for r in rows]
    values = [r[value_key] for r in rows]
    colors = [color_for(r["key"], picks) for r in rows]

    y_pos = list(range(len(rows)))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="none", height=0.72)
    ax.invert_yaxis()
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=13, color=COLOR_TEXT)
    ax.set_xlim(xlim)
    ax.set_title(title, fontsize=18, color=COLOR_TEXT, loc="left", pad=14, weight="bold")
    ax.tick_params(axis="x", colors=COLOR_MUTED, labelsize=11)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_MUTED)
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            value + (xlim[1] - xlim[0]) * 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            ha="left",
            fontsize=12,
            color=COLOR_TEXT,
            weight="bold",
        )


def main():
    rows = load_rows()
    top_mrr = top_by(rows, "mrr", 10)
    top_disc = top_by(rows, "disc", 10)
    open_rows = [r for r in rows if r["key"] != GEMINI]
    picks = {r["key"] for r in top_by(open_rows, "mrr", NUM_PICKS)}

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 12), dpi=100, gridspec_kw={"hspace": 0.28}
    )
    fig.patch.set_facecolor("white")

    fig.suptitle(
        "Uzbek RAG embedding benchmark — top 10 of 25 models",
        fontsize=22,
        color=COLOR_TEXT,
        weight="bold",
        x=0.06,
        y=0.975,
        ha="left",
    )
    fig.text(
        0.06,
        0.938,
        "7,078 passages · 2,017 queries · NVIDIA RTX 5070",
        fontsize=13,
        color=COLOR_MUTED,
        ha="left",
    )

    draw_panel(ax_top, top_mrr, "mrr", "MRR  (general retrieval quality)", (0, 1.02), picks)
    draw_panel(
        ax_bot,
        top_disc,
        "disc",
        "Hard-negative Discrimination  (production-critical)",
        (0, 1.02),
        picks,
    )

    # Legend
    legend_y = 0.045
    fig.text(0.06, legend_y, "\u25A0", fontsize=18, color=COLOR_PICK, va="center")
    fig.text(0.085, legend_y, "Recommended open picks", fontsize=12, color=COLOR_TEXT, va="center")
    fig.text(0.33, legend_y, "\u25A0", fontsize=18, color=COLOR_GEMINI, va="center")
    fig.text(0.355, legend_y, "Gemini (paid API ceiling)", fontsize=12, color=COLOR_TEXT, va="center")
    fig.text(0.58, legend_y, "\u25A0", fontsize=18, color=COLOR_OTHER, va="center")
    fig.text(0.605, legend_y, "Other open models", fontsize=12, color=COLOR_TEXT, va="center")

    fig.subplots_adjust(left=0.26, right=0.96, top=0.9, bottom=0.09)
    fig.savefig(OUT, dpi=100, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
