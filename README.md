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
│   ├── returns.py          #   return / reward-to-go estimators (vanilla vs RTG)
│   └── train.py            #   REINFORCE training placeholder (see experiments/)
├── experiments/            # runnable experiment scripts
│   ├── train_reinforce.py  #   REINFORCE training (vanilla / rtg) + Adam step
│   └── plot_results.py     #   plots from CSV logs
├── tests/                  # verification notebooks & scripts
│   ├── smoke_test_highway.ipynb  # env + spec check, 5-episode baseline, one GIF per episode
│   └── rollout_test.py     # policy network shape validation (states, actions, log_probs, rewards)
├── docs/                   # written deliverables
│   ├── notation.md         #   1-page notation cheat sheet
│   ├── derivation.md       #   policy gradient derivation (M2)
│   └── proofs.md           #   causality & baseline lemmas (M2)
├── results/                # experiment outputs
│   ├── logs/               #   CSV training logs (returns, grad norms, seeds)
│   ├── figures/            #   learning curves, variance comparison
│   └── videos/             #   untrained vs. trained agent clips
├── slides/                 # presentation deck parts (assembled Week 5)
└── REPORT.md               # final write-up (Week 5)
```

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

## Reproduction Commands

*To be filled in during Week 5 — one command per figure/table:*

```bash
# 1. Train (single seed, vanilla REINFORCE)
# 2. Train all variants x 3 seeds
# 3. Regenerate figures from CSV logs
```

## Weekly Progress

| Week | Goal | Status |
|---|---|---|
| 1 | Foundations: repo skeleton, requirements, notation sheet, env verified | ✅ scaffold + smoke test + GIFs |
| 2 | Core derivations, policy network, first architecture draft | ✅ policy.py, returns.py, train_reinforce.py |
| 3 | REINFORCE implementation & reward-to-go | ⏳ in progress (running test training run) |
| 3 | REINFORCE implementation & reward-to-go | ⬜ |
| 4 | Variance reduction experiments & media recording | ⬜ |
| 5 | Integration, slide polish & mock defense | ⬜ |

## Team

| Member | Role | Responsibilities |
|---|---|---|
| M1 | Lead | Repo, network code, training loop, experiments, final assembly |
| M2 | Math | Derivations, proofs, notation consistency, limitations slides |
| M3 | Foundations / Experiments | Env setup, baseline runs, plotting, results slides |
