"""Plots from the CSV experiment logs (tasks 44, 45, 46).

Produces three figures in `results/figures/`:

    learning_curves.png        task 45 — mean return vs episode, ± std across seeds
    variance_comparison.png    task 46 — spread of the returns per variant
    gradient_norms.png         task 41 — |grad| per episode, per variant

The two required plots (45, 46) and the one the gradient-norm logging
supports (41) all read from `results/logs/reinforce_{seed}_{mode}.csv`, so no
re-training is needed to regenerate them.

Usage
-----
    python experiments/plot_results.py
    python experiments/plot_results.py --outdir results/figures

Design notes
------------
* Curves are smoothed with a centred moving average for readability, and the
  per-episode values are drawn faintly underneath so the smoothing is visible
  rather than implied. With 30 episodes and 3 seeds the raw mean is very
  noisy; showing only the smooth line would overstate how clean the result is.
* The std band is the sample standard deviation **across seeds** at each
  episode index, which is what task 45 asks for. It is not the std across
  episodes within one run — those answer different questions.
* Episode counts differ between runs only if a run was cut short, so the
  curves are truncated to the shortest log before averaging.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless: no display needed
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "results" / "logs"

MODES = ("vanilla", "rtg", "baseline")

# Which variants the figures actually draw. Set from --modes on the command
# line; defaults to all three. Narrowing it matters when only some variants
# have a full 300-episode run: plotting a 30-episode log next to a 300-episode
# one draws a comparison that does not exist.
MODES_TO_PLOT = MODES

LABELS = {
    "vanilla": "Vanilla REINFORCE",
    "rtg": "Reward-to-go",
    "baseline": "Reward-to-go + learned baseline",
}
COLOURS = {
    "vanilla": "#C0504D",
    "rtg": "#4F81BD",
    "baseline": "#26A9E0",
}


# --------------------------------------------------------------------- io
def load(seed: int, mode: str) -> dict[str, np.ndarray]:
    """Load one run's CSV into column arrays. Missing file -> empty arrays."""
    path = LOG_DIR / f"reinforce_{seed}_{mode}.csv"
    if not path.exists():
        return {}
    cols: dict[str, list[float]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key in ("episode", "total_return", "steps", "loss", "grad_norm"):
                raw = row.get(key, "")
                if raw not in ("", None):
                    cols[key].append(float(raw))
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def discover_seeds(modes=MODES) -> list[int]:
    seeds = set()
    for p in LOG_DIR.glob("reinforce_*_*.csv"):
        parts = p.stem.split("_")
        if len(parts) >= 3 and parts[1].isdigit():
            seeds.add(int(parts[1]))
    return sorted(seeds)


def stack_runs(seeds: list[int], mode: str, column: str):
    """Return (episodes, matrix) where rows are seeds, truncated to min length.

    Truncating to the shortest log is what makes `np.std(axis=0)` meaningful:
    seeds that ran a different number of episodes would otherwise pad with
    NaN and make the band collapse at the end.
    """
    runs = [load(s, mode) for s in seeds]
    runs = [r for r in runs if r and column in r]
    if not runs:
        return None, None
    n = min(len(r[column]) for r in runs)
    if n == 0:
        return None, None
    mat = np.vstack([r[column][:n] for r in runs])
    return np.arange(n), mat


def smooth(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Centred moving average, edge-padded so the ends are not short."""
    if len(x) < window or window <= 1:
        return x
    pad = window // 2
    padded = np.pad(x, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: len(x)]


# ------------------------------------------------------------------ plots
def plot_learning_curves(seeds: list[int], outdir: Path) -> Path | None:
    """Task 45: mean return vs episode with a shaded ± std band across seeds."""
    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=150)

    plotted = False
    for mode in MODES_TO_PLOT:
        eps, mat = stack_runs(seeds, mode, "total_return")
        if eps is None:
            continue
        mean = mat.mean(axis=0)
        std = mat.std(axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(mean)

        # raw per-seed traces, faint — so the reader can see the noise that
        # the smoothing removes rather than having to trust it
        for row in mat:
            ax.plot(eps, row, color=COLOURS[mode], alpha=0.13, linewidth=0.9)

        ax.plot(eps, smooth(mean), color=COLOURS[mode], linewidth=2.2,
                label=f"{LABELS[mode]} (n={mat.shape[0]})")
        ax.fill_between(eps, smooth(mean - std), smooth(mean + std),
                        color=COLOURS[mode], alpha=0.18, linewidth=0)
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return (undiscounted)")
    ax.set_title("REINFORCE on highway-v0 — mean return across seeds\n"
                 f"seeds = {seeds}, shaded band = ±1 std across seeds, "
                 "line = centred 5-episode mean")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = outdir / "learning_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_variance_comparison(seeds: list[int], outdir: Path) -> Path | None:
    """The spread of RETURNS across and within runs.

    This is NOT the variance of the gradient estimator. An earlier version
    wrote this to `variance_comparison.png`, the same filename that
    `measure_variance.py` uses for the gradient measurement, so whichever
    script ran last silently replaced the other's figure. They answer
    different questions and now have different filenames.

    Two panels, because 'variance of the return' can mean two different
    things and conflating them is what makes this plot misleading:

    left   the spread ACROSS SEEDS at each episode (run-to-run variance —
           how much the result depends on the lucky initialisation)
    right  the per-episode spread WITHIN a run, averaged (step-to-step
           variance — how noisy a single training run looks)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=150)

    any_data = False
    across_means: dict[str, float] = {}
    within_means: dict[str, float] = {}

    for mode in MODES_TO_PLOT:
        eps, mat = stack_runs(seeds, mode, "total_return")
        if eps is None:
            continue
        any_data = True

        if mat.shape[0] > 1:
            across = mat.std(axis=0, ddof=1)
            ax1.plot(eps, smooth(across), color=COLOURS[mode], linewidth=2.0,
                     label=LABELS[mode])
            across_means[mode] = float(across.mean())

        within = mat.std(axis=1, ddof=1)
        ax2.plot(eps, smooth(mat.mean(axis=0)), color=COLOURS[mode],
                 linewidth=2.0, label=LABELS[mode])
        ax2.fill_between(eps,
                         smooth(mat.mean(axis=0) - within.mean()),
                         smooth(mat.mean(axis=0) + within.mean()),
                         color=COLOURS[mode], alpha=0.15, linewidth=0)
        within_means[mode] = float(within.mean())

    if not any_data:
        plt.close(fig)
        return None

    ax1.set_title("Spread ACROSS seeds (run-to-run variance)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("std of return across seeds")
    ax1.legend(frameon=False, fontsize=9)

    ax2.set_title("Spread WITHIN a run (mean ± 1 std over episodes)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode return")

    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Spread of the episode return, three variants", fontsize=11)
    fig.tight_layout()

    out = outdir / "return_spread.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    # numbers in the console too — a reader should not have to read a chart
    # to get the one comparison the task asks for
    print("\n  mean std across seeds   (lower = more reproducible):")
    for mode, v in across_means.items():
        print(f"    {mode:<10} {v:6.3f}")
    print("\n  mean std within a run   (lower = smoother single run):")
    for mode, v in within_means.items():
        print(f"    {mode:<10} {v:6.3f}")
    return out


def plot_gradient_norms(seeds: list[int], outdir: Path) -> Path | None:
    """Task 41: the gradient norm that the CSV logging added."""
    fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=150)

    plotted = False
    for mode in MODES_TO_PLOT:
        eps, mat = stack_runs(seeds, mode, "grad_norm")
        if eps is None:
            continue
        mean = mat.mean(axis=0)
        std = mat.std(axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(mean)
        ax.plot(eps, smooth(mean), color=COLOURS[mode], linewidth=2.0,
                label=LABELS[mode])
        ax.fill_between(eps, smooth(mean - std), smooth(mean + std),
                        color=COLOURS[mode], alpha=0.15, linewidth=0)
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_xlabel("Episode")
    ax.set_ylabel("‖∇_θ L‖₂  (before the optimiser step)")
    ax.set_title("Policy-gradient magnitude per episode")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = outdir / "gradient_norms.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    global MODES_TO_PLOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "results" / "figures"))
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES,
                    help="variants to draw; pass only those with equal-length logs")
    args = ap.parse_args()

    MODES_TO_PLOT = tuple(args.modes)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds or discover_seeds(MODES_TO_PLOT)
    if not seeds:
        print(f"no CSVs in {LOG_DIR} — run experiments/run_batch.py first")
        return 1

    print(f"seeds found: {seeds}")
    print(f"logs       : {LOG_DIR}")
    print(f"variants   : {MODES_TO_PLOT}")

    made = []
    for fn, title in ((plot_learning_curves, "learning_curves.png"),
                      (plot_variance_comparison, "return_spread.png"),
                      (plot_gradient_norms, "gradient_norms.png")):
        out = fn(seeds, outdir)
        if out is None:
            print(f"  skip {title} — no data")
        else:
            made.append(out)
            print(f"  wrote {out.relative_to(ROOT)}  "
                  f"({out.stat().st_size/1024:.0f} KB)")

    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
