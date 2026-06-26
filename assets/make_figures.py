"""Regenerate the README charts from a deterministic StaleBench run.

    python assets/make_figures.py     # writes assets/*.png

Numbers come straight from the mock-provider benchmark (seeded => reproducible),
so the figures always match what `aml bench` prints.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agent_memory_lab.stalebench.runner import run_benchmark

HERE = os.path.dirname(os.path.abspath(__file__))
BACKENDS = ["append-only", "append-recency", "key-overwrite", "bitemporal"]


def main() -> None:
    rows = run_benchmark(BACKENDS, provider="mock", n_episodes=30)
    names = [r["backend"] for r in rows]

    # --- Figure 1: correctness below the final score ----------------------- #
    metrics = [("staleness_rate", "staleness (lower=better)", "#e4572e"),
               ("update_em", "update-EM", "#3a86ff"),
               ("history_recall", "history recall", "#2a9d8f")]
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for i, (key, label, color) in enumerate(metrics):
        vals = [r[key] * 100 for r in rows]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=label, color=color)
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10, ha="right")
    ax.set_ylabel("%")
    ax.set_ylim(0, 112)
    ax.set_title("StaleBench — what end-to-end EM hides (mock, 30 episodes)")
    ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "staleness_benchmark.png"), dpi=130)

    # --- Figure 2: cost vs staleness, bubble = items stored ---------------- #
    fig2, ax2 = plt.subplots(figsize=(8.4, 4.6))
    palette = {"append-only": "#e4572e", "append-recency": "#f4a259",
               "key-overwrite": "#3a86ff", "bitemporal": "#2a9d8f"}
    for i, r in enumerate(rows):
        cost, stale, store = r["avg_scanned"], r["staleness_rate"] * 100, r["avg_store_size"]
        ax2.scatter(cost, stale, s=store * 34, alpha=0.65,
                    color=palette.get(r["backend"], "#888"), edgecolor="black", zorder=3)
        dy = 12 if i % 2 == 0 else -16
        ax2.annotate(f"{r['backend']}\n(stores {store:.0f})", (cost, stale),
                     textcoords="offset points", xytext=(8, dy), fontsize=8.5)
    ax2.set_xlabel("items scanned per query  →  retrieval cost")
    ax2.set_ylabel("staleness rate (%)")
    ax2.set_xlim(-1, 18)
    ax2.set_ylim(-8, 65)
    ax2.set_title("Cost vs staleness — bubble area = items kept in memory")
    ax2.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(os.path.join(HERE, "cost_vs_correctness.png"), dpi=130)
    print("wrote assets/staleness_benchmark.png, assets/cost_vs_correctness.png")


if __name__ == "__main__":
    main()
