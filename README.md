# Policy Gradient for Reinforcement Learning

A from-scratch implementation and mathematical derivation of the
**REINFORCE** policy gradient algorithm, evaluated on
**highway-env's `highway-v0`** (Gymnasium): a lane-keeping
driving task with 5 discrete driving actions. The project pairs
working PyTorch code with step-by-step analytical proofs and a
comparative variance-reduction study (vanilla return vs.
reward-to-go vs. baseline).

## Repository Structure

```
.
├── README.md               # this file: overview + reproduction commands
├── requirements.txt        # torch, highway-env, matplotlib, numpy, imageio, jupyterlab
├── src/                    # core library code
│   ├── __init__.py
│   ├── policy.py           #   policy network (25 -> 128 ReLU -> 128 ReLU -> 5 Softmax) + Categorical act
│   └── returns.py          #   return / reward-to-go estimators (vanilla vs RTG)
├── experiments/            # runnable experiment scripts
│   └── train_reinforce.py  #   REINFORCE training (vanilla / rtg) + Adam step
├── tests/                  # verification notebooks & scripts
│   ├── smoke_test_highway.ipynb      # Week 1: env + spec check, 5-episode baseline, one GIF per episode
│   ├── reinforce_implementation.ipynb  # Week 3: tasks 27-30, return/loss/Adam — verified numerically
│   ├── build_notebook.py             #   regenerates reinforce_implementation.ipynb from source
│   └── rollout_test.py               # policy network shape validation
├── docs/                   # written deliverables
│   ├── notation.md         #   1-page notation cheat sheet
│   └── highway_spec.md     #   Kinematics obs matrix + action definitions
└── results/                # experiment outputs
    ├── logs/               #   CSV training logs (returns, per seed)
    └── videos/             #   untrained agent baseline clips
```

> **Slide material lives outside this repo.**
> The deck, its builder, rendered previews and LaTeX equation cache are in
> `../Slide/`. Nothing here depends on them, so the repo builds and runs on a
> clean checkout.

## Docs
- [Notation cheat sheet](docs/notation.md)
- [highway-v0 spec table](docs/highway_spec.md)

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute tests/smoke_test_highway.ipynb
# ...or open it interactively: jupyter lab tests/smoke_test_highway.ipynb
```

The smoke-test notebook prints the highway observation/action spaces,
records the lengths and returns of 5 random-policy episodes, and saves
one animated GIF per episode to `results/videos/` — a passing run means
the local environment is ready.

### Windows: one environment variable

`torch` and `highway-env` each ship their own Intel OpenMP runtime. Importing
both into one process aborts with `OMP: Error #15` and the kernel dies with no
Python traceback. Set this before launching Jupyter:

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

`tests/reinforce_implementation.ipynb` sets it inside its first cell, so a
top-to-bottom run works without shell configuration; the smoke test still
needs it in the environment.

## Reproduction Commands

```bash
# 1. Environment smoke test (Week 1): spec check + 5 baseline episodes + GIFs
jupyter nbconvert --to notebook --execute tests/smoke_test_highway.ipynb

# 2. Implementation walkthrough (Week 3, tasks 27-30)
#    Verifies returns, the REINFORCE loss gradient and one Adam step.
jupyter nbconvert --to notebook --execute tests/reinforce_implementation.ipynb

# 3. Policy-network shape check
python tests/rollout_test.py

# 4. Train one variant (writes results/logs/reinforce_<seed>_<mode>.csv)
python experiments/train_reinforce.py --mode rtg --seed 0 --epochs 30
```

Regenerate the Week 3 notebook after editing `tests/build_notebook.py`:

```bash
python tests/build_notebook.py
```

## Weekly Progress

| Week | Goal | Status |
|---|---|---|
| 1 | Foundations: repo skeleton, requirements, notation sheet, env verified | ✅ scaffold + smoke test + GIFs |
| 2 | Core derivations, policy network, first architecture draft | ✅ policy.py, returns.py |
| 3 | REINFORCE implementation & reward-to-go | ✅ returns + loss + Adam, verified in `reinforce_implementation.ipynb` |
| 4 | Variance reduction experiments & media recording | ⬜ 3-seed batch not run yet |
| 5 | Integration, slide polish & mock defense | ⬜ |

## Team

| Member | Role | Responsibilities |
|---|---|---|
| M1 | Lead | Repo, network code, training loop, experiments, final assembly |
| M2 | Math | Derivations, proofs, notation consistency, limitations slides |
| M3 | Foundations / Experiments | Env setup, baseline runs, plotting, results slides |
