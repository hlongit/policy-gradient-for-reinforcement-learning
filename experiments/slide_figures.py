"""Slide-ready figures: two variants, three seeds, one message per panel.

The analysis figures in plot_results.py are deliberately busy — they show
every seed's raw trace and a std band, which is what you want when checking
your own work. A projected slide needs the opposite: one thick line per
variant, the raw traces faint behind it for honesty, and the final numbers
printed where the eye lands.

Writes:
    results/figures/slide_learning_curve.png
    results/figures/slide_baseline_ablation.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "results" / "logs"
FIG_DIR = ROOT / "results" / "figures"

COLOUR = {"vanilla": "#MISSING"}  # replaced below
COLOUR = {"vanilla": "#C0504D", "rtg": "#4F81BD"}
LABEL = {"vanilla": "Vanilla REINFORCE\nw = R(τ)",
         "rtg": "Reward-to-go\nw = Gₜ"}
SEEDS = [0, 1, 2]


def load(seed: int, mode: str) -> np.ndarray:
    path = LOG_DIR / f"reinforce_{seed}_{mode}.csv"
    if not path.exists():
        return np.array([])
    with open(path, encoding="utf-8") as f:
        return np.array([float(r["total_return"]) for r in csv.DictReader(f)])


def smooth(x: np.ndarray, w: int = 9) -> np.ndarray:
    if len(x) < w:
        return x
    k = np.ones(w) / w
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, k, mode="valid")[:len(x)]


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------ learning curve
    fig, ax = plt.subplots(figsize=(9.6, 5.0), dpi=200)

    for mode in ("vanilla", "rtg"):
        runs = [load(s, mode) for s in SEEDS]
        runs = [r for r in runs if len(r) >= 300]
        if not runs:
            continue
        n = min(len(r) for r in runs)
        mat = np.stack([r[:n] for r in runs])

        # raw traces: faint, so the reader sees the real noise
        for row in mat:
            ax.plot(np.arange(n), smooth(row), color=COLOUR[mode],
                    linewidth=0.7, alpha=0.22, zorder=1)
        # seed mean, thicker
        mean = mat.mean(axis=0)
        ax.plot(np.arange(n), smooth(mean, 15), color=COLOUR[mode],
                linewidth=3.0, zorder=3, label=LABEL[mode])

        # final value, printed
        final = mean[-20:].mean()
        ax.annotate(f"{final:.1f}",
                    xy=(n - 1, smooth(mean, 15)[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    color=COLOUR[mode], fontsize=13, fontweight="bold",
                    va="center")

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Episode return", fontsize=11)
    ax.set_title("Same update rule, different weight  —  highway-v0, 3 seeds, 300 episodes",
                 fontsize=12.5, pad=14)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.22, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 300)
    fig.tight_layout()
    out = FIG_DIR / "slide_learning_curve.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))

    # ------------------------------------------------ baseline ablation
    import json
    bq_path = FIG_DIR / "baseline_quality.json"
    if bq_path.exists():
        bq = json.loads(bq_path.read_text(encoding="utf-8"))
        pv = bq["proj_var"]
        base = pv["vanilla"]

        order = [("vanilla", "Vanilla\nw = R(τ)"),
                 ("rtg", "RTG\nw = Gₜ"),
                 ("rtg-const", "RTG − hằng số\nb = E[G]"),
                 ("rtg-linear", "RTG − critic tuyến tính\nb = V_φ(s)")]
        names = [k for k, _ in order]
        labels = [l for _, l in order]
        vals = [pv[k] / base for k in names]
        gains = [base / pv[k] for k in names]

        fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=200)
        xs = np.arange(len(names))
        bars = ax.bar(xs, vals, width=0.62,
                      color=["#C0504D", "#4F81BD", "#8FAADC", "#26A9E0"])
        for b, v, g in zip(bars, vals, gains):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                    f"{v:.2f}×\ngiảm {g:.2f}× số mẫu",
                    ha="center", va="bottom", fontsize=9.5)
        ax.axhline(1.0, color="#888", linewidth=1.0, linestyle="--", zorder=0)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_ylabel("Phương sai gradient\n(tương đối so với vanilla)", fontsize=10.5)
        ax.set_title("Baseline chỉ giúp khi nó dự đoán tốt\n"
                     f"(critic tuyến tính giải thích R² = {bq['r2_linear']:.2f} phương sai của Gₜ)",
                     fontsize=11.5, pad=12)
        ax.grid(True, axis="y", alpha=0.22, linestyle=":")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(vals) * 1.38)
        fig.tight_layout()
        out = FIG_DIR / "slide_baseline_ablation.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
