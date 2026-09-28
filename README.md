# Policy Gradient for Reinforcement Learning

A from-scratch implementation and mathematical derivation of the
**REINFORCE** policy gradient algorithm, evaluated on **CartPole-v1**
(Gymnasium). The project pairs working PyTorch code with step-by-step
analytical proofs and a comparative variance-reduction study
(vanilla return vs. reward-to-go vs. baseline).

## Repository Structure

```
.
├── README.md               # this file: overview + reproduction commands
├── requirements.txt        # torch, gymnasium, matplotlib, numpy
├── src/                    # core library code
│   ├── policy.py           #   policy network (MLP + softmax head)
│   ├── returns.py          #   return / reward-to-go / advantage computations
│   └── train.py            #   REINFORCE training loop
├── experiments/            # runnable experiment scripts
│   ├── train_reinforce.py  #   training entry point (vanilla / rtg / baseline)
│   └── plot_results.py     #   plots from CSV logs
├── tests/                  # verification notebooks
│   └── smoke_test_cartpole.ipynb  # env + spec sanity check (5 random episodes)
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

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute tests/smoke_test_cartpole.ipynb
# ...or open it interactively: jupyter lab tests/smoke_test_cartpole.ipynb
```

The smoke-test notebook prints the CartPole observation/action spaces and the
episode lengths of 5 random-policy episodes — a passing run means the
local environment is ready.

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
| 1 | Foundations: repo skeleton, requirements, notation sheet, env verified | ✅ scaffold in place |
| 2 | Core derivations, policy network, first architecture draft | ⬜ |
| 3 | REINFORCE implementation & reward-to-go | ⬜ |
| 4 | Variance reduction experiments & media recording | ⬜ |
| 5 | Integration, slide polish & mock defense | ⬜ |

## Team

| Member | Role | Responsibilities |
|---|---|---|
| M1 | Lead | Repo, network code, training loop, experiments, final assembly |
| M2 | Math | Derivations, proofs, notation consistency, limitations slides |
| M3 | Foundations / Experiments | Env setup, baseline runs, plotting, results slides |
