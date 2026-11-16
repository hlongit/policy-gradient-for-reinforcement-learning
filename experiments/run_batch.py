"""Batch runner — 3 seeds x {vanilla, rtg, baseline} (task 40).

Runs every (seed, variant) combination in its own subprocess so a crash in one
run cannot corrupt the others, then prints a summary table of the final
returns. Each subprocess writes its own CSV, so the batch is resumable: a
combination is skipped when its CSV already has the expected number of rows.

Usage
-----
    python experiments/run_batch.py                    # 3 seeds x 3 variants
    python experiments/run_batch.py --epochs 30
    python experiments/run_batch.py --seeds 0 1 2 --force

Why subprocesses instead of calling train_one_mode in a loop
------------------------------------------------------------
torch and highway-env both hold global state (RNG, the OMP runtime, and the
Gymnasium registry). Running each combination in a fresh interpreter makes the
seeds reproducible regardless of what a previous run left behind, and matches
how the commands in the README are executed one at a time.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "results" / "logs"
TRAIN = ROOT / "experiments" / "train_reinforce.py"

MODES = ("vanilla", "rtg", "baseline")


def rows_in(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)     # minus the header


def final_return(path: Path) -> tuple[float, float]:
    """(mean of last 10 episodes, mean of all episodes) from a CSV log."""
    if not path.exists():
        return float("nan"), float("nan")
    with open(path, encoding="utf-8") as f:
        vals = [float(r["total_return"]) for r in csv.DictReader(f)]
    if not vals:
        return float("nan"), float("nan")
    tail = vals[-10:]
    return sum(tail) / len(tail), sum(vals) / len(vals)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--force", action="store_true",
                    help="re-run even if the CSV already has enough rows")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    runs = [(s, m) for s in args.seeds for m in args.modes]
    print(f"batch: {len(runs)} runs "
          f"({len(args.seeds)} seeds x {len(args.modes)} variants), "
          f"{args.epochs} episodes each\n")

    started = time.time()
    failed: list[tuple[int, str]] = []

    for i, (seed, mode) in enumerate(runs, 1):
        csv_path = LOG_DIR / f"reinforce_{seed}_{mode}.csv"
        done = rows_in(csv_path)

        if not args.force and done >= args.epochs:
            print(f"[{i}/{len(runs)}] {mode:<9} seed={seed}  "
                  f"skip ({done} rows already logged)")
            continue

        print(f"[{i}/{len(runs)}] {mode:<9} seed={seed}  running...", flush=True)
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(TRAIN),
             "--mode", mode, "--seed", str(seed), "--epochs", str(args.epochs)],
            cwd=str(ROOT), capture_output=True, text=True,
        )
        dt = time.time() - t0

        if proc.returncode != 0:
            failed.append((seed, mode))
            print(f"    FAILED after {dt:.1f}s")
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
            for line in tail:
                print("    " + line)
            continue

        tail_mean, all_mean = final_return(csv_path)
        print(f"    ok  {dt:5.1f}s   last10 {tail_mean:6.2f}   "
              f"all {all_mean:6.2f}")

    total = time.time() - started
    print(f"\nbatch finished in {total/60:.1f} min")

    # ---------------------------------------------------------------- summary
    print()
    hdr = f"{'variant':<10}" + "".join(f"{f'seed {s}':>10}" for s in args.seeds)
    print("mean return over the last 10 episodes")
    print(hdr)
    print("-" * len(hdr))
    for mode in args.modes:
        cells = []
        for s in args.seeds:
            tail, _ = final_return(LOG_DIR / f"reinforce_{s}_{mode}.csv")
            cells.append(f"{tail:>10.2f}" if tail == tail else f"{'--':>10}")
        print(f"{mode:<10}" + "".join(cells))

    if failed:
        print(f"\n{len(failed)} run(s) failed: {failed}")
        return 1
    print("\nall runs completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
