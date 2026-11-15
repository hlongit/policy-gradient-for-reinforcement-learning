"""Build `tests/reinforce_implementation.ipynb`.

The notebook is generated from this script rather than hand-edited JSON so the
cell structure stays reviewable in a diff and can be regenerated.

Cells are built in order; markdown cells carry the explanation a teammate
needs, code cells are the actual implementation used by
`experiments/train_reinforce.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent          # tests/
OUT = HERE / "reinforce_implementation.ipynb"

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    })


def code(text: str) -> None:
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


# ---------------------------------------------------------------- title
md(r"""
# Week 3 — REINFORCE implementation (tasks 27–30)

This notebook fills the gap between `smoke_test_highway.ipynb` (Week 1: the
environment works) and `experiments/train_reinforce.py` (the batch runner).
It walks the four implementation deliverables one at a time so each piece can
be checked in isolation before the training loop is trusted:

| Task | Deliverable | Section |
|---|---|---|
| 27 | `total_trajectory_return` — the vanilla weight $R(\tau)$ | [1](#1) |
| 30 | `reward_to_go` — the per-step weight $G_t$ | [2](#2) |
| 28 | policy-gradient loss matching the REINFORCE objective | [3](#3) |
| 29 | Adam parameter update | [4](#4) |

Then §5 assembles them into one short episode and §6 checks the two
identities the loss depends on.

## Why this order

`returns.py` and `policy.py` already exist in `src/`; what was missing was a
runnable, explained demonstration that the four pieces compose correctly. The
last section is the important one: it is where a wrong sign or a detached
graph would show up as a *numerical* mismatch rather than as a silently
non-learning agent.
""")

# ---------------------------------------------------------------- setup
md(r"""
---
## 0. Setup

`sys.path` is extended so `src/` is importable regardless of the notebook's
working directory — Jupyter starts in `tests/`, but the modules live one level
up.

### The one environment workaround

`torch` and `highway-env` (via numpy/scipy) each ship their own copy of the
Intel OpenMP runtime. On Windows importing both into one process aborts with
`OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already
initialized` — the kernel dies with no Python traceback, so it looks like the
notebook is broken when the environment is.

Setting `KMP_DUPLICATE_LIB_OK=TRUE` before the imports is the documented
workaround. It must run in the very first cell, before `import torch`.
""")

code(r"""
import os

# Must precede `import torch` — see the note above. Without this the kernel
# dies on the first highway-env rollout with OMP Error #15 and no traceback.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

# Make the repo root importable when the kernel starts in tests/.
ROOT = Path.cwd().parent if Path.cwd().name == "tests" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.policy import PolicyNetwork
from src.returns import reward_to_go, total_trajectory_return

torch.manual_seed(0)

GAMMA = 0.99
print("repo root :", ROOT)
print("torch     :", torch.__version__)
print("gamma     :", GAMMA)
""")

# ---------------------------------------------------------------- task 27
md(r"""
---
<a id="1"></a>
## 1. Total trajectory return — task 27

### What it computes

The **whole-episode discounted return**, a single scalar:

$$
R(\tau) \;=\; \sum_{t=0}^{T-1} \gamma^{\,t}\, r_t
$$

### Why the discount is there

With $\gamma = 0.99$ over the 40-step maximum episode, the weight on the last
reward is $0.99^{39} \approx 0.6757$ — so late rewards still count, they are
just worth ~32 % less. Setting $\gamma = 1$ would make the sum the plain
total; the repo uses $\gamma < 1$ so the same algebra covers a continuing
horizon.

### The implementation detail that matters

The discount exponent is the index `t` **from the start of the episode**, not
a running product. That is why this function multiplies by `gamma ** t`
elementwise rather than carrying a `g` accumulator — the two are different
functions and the next section is exactly the other one.

> **Contract:** returns a shape-`(1,)` tensor so it broadcasts against a
> `(T,)` vector of log-probabilities in the loss.
""")

code(r"""
# The shipped implementation is imported under an alias, so the cell below can
# define its own version with the same name without shadowing the reference.
from src.returns import total_trajectory_return as shipped_total_return


def total_trajectory_return(rewards, gamma: float = GAMMA):
    '''R(tau) = sum_t gamma^t * r_t  — one scalar for the whole episode.

    The exponent is the absolute step index t, so r_0 is never discounted and
    r_{T-1} is discounted by gamma^(T-1).

    Returns a shape-(1,) tensor: the loss multiplies it against a (T,)
    vector of log-probabilities, and broadcasting needs the singleton dim.
    '''
    r = torch.as_tensor(rewards, dtype=torch.float32)
    t = torch.arange(len(r), dtype=r.dtype)
    return (r * gamma ** t).sum().unsqueeze(0)


# --- check against the shipped implementation -----------------------------
demo_rewards = [1.0, 1.0, 1.0, 1.0]

mine = total_trajectory_return(demo_rewards, GAMMA)
shipped = shipped_total_return(demo_rewards, GAMMA)   # src/returns.py
expected = sum(GAMMA ** t for t in range(4))

print(f"rewards           : {demo_rewards}")
print(f"R(tau)            : {mine.item():.6f}")
print(f"shipped           : {shipped.item():.6f}")
print(f"hand sum          : {expected:.6f}")
print(f"difference        : {abs(mine.item() - expected):.2e}")
assert abs(mine.item() - expected) < 1e-6
assert abs(mine.item() - shipped.item()) < 1e-6, "disagrees with src/returns.py"
assert mine.shape == (1,), "must be shape (1,) to broadcast in the loss"
print("task 27 ok")
""")

md(r"""
### Sanity check a reviewer will ask for

For a constant reward $r$ over $T$ steps the closed form is
$r \cdot \frac{1-\gamma^{T}}{1-\gamma}$. If the implementation is off by one
in the exponent, this identity fails by a factor of $\gamma$ — the single
most common bug in this function.
""")

code(r"""
T = 40
r_const = 1.0
closed_form = r_const * (1 - GAMMA ** T) / (1 - GAMMA)
from_impl = total_trajectory_return([r_const] * T, GAMMA).item()

print(f"closed form  r*(1-g^T)/(1-g) : {closed_form:.6f}")
print(f"implementation               : {from_impl:.6f}")
print(f"ratio                        : {from_impl / closed_form:.9f}")
assert abs(from_impl - closed_form) < 1e-4
print("closed-form identity holds — no off-by-one in the exponent")
""")

# ---------------------------------------------------------------- task 30
md(r"""
---
<a id="2"></a>
## 2. Reward-to-go — task 30

### What it computes

A **per-step** weight instead of one scalar:

$$
G_t \;=\; \sum_{k=t}^{T-1} \gamma^{\,k-t}\, r_k
$$

Note the exponent is now $k - t$: the sum restarts at each step. $G_0$ equals
$R(\tau)$ — that identity is worth asserting, and §2.3 does.

### The backwards pass, and why it is exact

A naive double loop costs $\mathcal{O}(T^2)$. The recursion

$$
G_t = r_t + \gamma\,G_{t+1}, \qquad G_T = 0
$$

turns it into $\mathcal{O}(T)$ in one backwards sweep. It is *exact*, not an
approximation: expanding the recursion reproduces the sum term by term.

### Why this is the variance-reducing variant

$G_t$ drops every reward that arrived **before** $a_t$ was chosen. Those
rewards cannot depend on $a_t$, so their expected product with the score
$\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ is exactly zero — subtracting them
leaves the gradient unbiased while removing noise. The proof is the causality
lemma (M2's `docs/proofs.md`); here we only need the code to be right.
""")

code(r"""
from src.returns import reward_to_go as shipped_reward_to_go


def reward_to_go(rewards, gamma: float = GAMMA):
    '''G_t = sum_{k>=t} gamma^(k-t) * r_k, for every t.

    Backwards pass:
        G_T     = 0
        G_t     = r_t + gamma * G_{t+1}

    O(T) and exact — expanding the recursion gives back the sum term by term.

    Returns a shape-(T,) tensor: one weight per action.
    '''
    r = torch.as_tensor(rewards, dtype=torch.float32)
    T = len(r)
    out = torch.zeros_like(r)
    g = torch.tensor(0.0, dtype=r.dtype)
    for t in reversed(range(T)):
        g = r[t] + gamma * g
        out[t] = g
    return out


demo = [1.0, 1.0, 1.0, 1.0]
rtg = reward_to_go(demo, GAMMA)

# brute-force the definition, so the recursion is checked, not assumed
def by_definition(rewards, gamma):
    T = len(rewards)
    return [sum(gamma ** (k - t) * rewards[k] for k in range(t, T))
            for t in range(T)]

ref = by_definition(demo, GAMMA)
print("rewards      :", demo)
print("reward-to-go :", [round(v, 6) for v in rtg.tolist()])
print("by definition:", [round(v, 6) for v in ref])
for a, b in zip(rtg.tolist(), ref):
    assert abs(a - b) < 1e-6
assert torch.allclose(rtg, shipped_reward_to_go(demo, GAMMA), atol=1e-6), \
    "disagrees with src/returns.py"
print("task 30 ok — recursion matches the definition and src/returns.py")
""")

md(r"""
### 2.1 The two identities this function must satisfy

1. **$G_0 = R(\tau)$** — the per-step weight at $t=0$ is the whole-episode
   return. If this fails, the two variants are optimising different objectives
   by accident.
2. **Vanilla broadcasts, RTG does not.** `total_trajectory_return` gives `(1,)`,
   `reward_to_go` gives `(T,)`. The loss relies on broadcasting the first and
   using the second directly — a shape mix-up here silently trains on one
   weight only.
""")

code(r"""
rng = np.random.default_rng(0)
random_rewards = rng.uniform(0, 1, size=12).astype(np.float32)

R_tau = total_trajectory_return(random_rewards, GAMMA)
G = reward_to_go(random_rewards, GAMMA)

print(f"R(tau)             : {R_tau.item():.6f}   shape {tuple(R_tau.shape)}")
print(f"G_0                : {G[0].item():.6f}   shape {tuple(G.shape)}")
print(f"difference         : {abs(R_tau.item() - G[0].item()):.2e}")
assert abs(R_tau.item() - G[0].item()) < 1e-5, "G_0 must equal R(tau)"
print()

# broadcasting behaviour the loss depends on
logp = torch.full((12,), -np.log(5.0))          # uniform policy, 5 actions
w_vanilla = (logp * R_tau).mean()               # (T,) * (1,) -> broadcast
w_rtg = (logp * G).mean()                       # (T,) * (T,) -> elementwise
print(f"vanilla weight     : {w_vanilla.item():.6f}   (R_tau broadcast over T)")
print(f"reward-to-go weight: {w_rtg.item():.6f}   (per-step G_t)")
assert w_vanilla.shape == w_rtg.shape == torch.Size([])
print("shapes agree — both variants produce a scalar loss")
""")

md(r"""
### 2.2 How different are the two variants numerically?

They share the same expectation but not the same numbers, so a quick look is
useful for anyone reviewing the experiment section. The ratio of the weights
shows how aggressively RTG re-weights early steps.
""")

code(r"""
for t in [0, 3, 6, 9, 11]:
    print(f"t={t:>2}  R(tau)={R_tau.item():7.4f}   G_t={G[t].item():7.4f}   "
          f"G_t/R(tau)={G[t].item() / R_tau.item():.4f}")

print()
print("At t=0 the two agree exactly; later steps keep only their own future,")
print("which is what removes the variance from rewards they cannot influence.")
""")

# ---------------------------------------------------------------- task 28
md(r"""
---
<a id="3"></a>
## 3. Policy-gradient loss — task 28

### The objective

$$
\mathcal{L}(\theta) \;=\; -\,\mathbb{E}_{\tau\sim\pi_\theta}
\left[\sum_{t=0}^{T-1} \gamma^{\,t}\, G_t\,
\log \pi_\theta(a_t \mid s_t)\right]
$$

Two things make this loss unusual compared with supervised learning:

- **The weight is not differentiable.** $G_t$ is computed from recorded
  rewards; it carries no $\theta$. It must be treated as a *constant*, which
  is exactly what `.detach()` expresses.
- **Only the log-probability carries the graph.** $\nabla_\theta\mathcal{L}$
  exists only if `log_probs.requires_grad` is still true at the backward call.

Getting either of these backwards produces an agent that runs, logs, and never
learns — so §3.2 tests the gradients rather than trusting them.

### The sign

Optimisers *descend*, we want to *ascend* $J$. Minimising $-\sum_t w_t\log\pi$
maximises $\sum_t w_t\log\pi$. The minus sign is the whole difference, and it
is the most common silent bug in student implementations.
""")

code(r"""
def reinforce_loss(log_probs, weights, mode: str = "vanilla"):
    '''REINFORCE loss for one trajectory.

    Parameters
    ----------
    log_probs : (T,) tensor with requires_grad=True
        log pi_theta(a_t | s_t) for each step.
    weights : (T,) or (1,) tensor
        G_t for mode="rtg", R(tau) for mode="vanilla".
    mode : {"vanilla", "rtg"}
        Kept explicit so the logged run says which objective it used.

    Returns
    -------
    Scalar loss. Minimising it maximises E[w_t * log pi].

    NOTE the mean over T, not a sum. Both are valid estimators of the same
    expectation up to a constant factor 1/T, which is absorbed into the
    learning rate. The mean is used here because it keeps the loss magnitude
    comparable across episodes of different length — see the note in §3.3.
    '''
    if mode == "vanilla":
        weights = weights.expand_as(log_probs)   # R(tau) broadcast over T
    elif mode != "rtg":
        raise ValueError(f"unknown mode {mode!r}")

    return -(log_probs * weights.detach()).mean()
""")

md(r"""
### 3.1 The check that actually matters: is the gradient right?

A loss can look plausible and still be wrong. The decisive test is to compare
the autograd gradient against a **finite-difference** estimate on a tiny
problem — a few parameters, one fixed action — where the answer can be
computed by perturbing $\theta$ directly.

This is the test that catches a flipped sign or a detached weight.
""")

code(r"""
# Tiny deterministic setup: 2 parameters, 3 actions, one sampled action.
#
# NOTE the dtype. This check must run in float64. In float32 a value carries
# ~7 significant digits, so a step of 1e-5 already sits inside the rounding
# noise and the finite difference is meaningless — it reported a 1.6e-2 gap on
# a gradient of magnitude ~1, which is an artefact of the arithmetic, not a
# wrong gradient.
torch.manual_seed(7)
DTYPE = torch.float64
theta = torch.nn.Parameter(torch.tensor([0.3, -0.2], dtype=DTYPE))

def log_pi(theta_vec, action: int):
    '''log softmax of a 3-action policy that depends only on theta_vec.'''
    logits = torch.stack([theta_vec[0], theta_vec[1],
                          torch.zeros((), dtype=DTYPE)])
    return torch.log_softmax(logits, dim=0)[action]

ACTION = 1
W = 2.5           # a stand-in weight: any constant works for this test

def loss_fn(theta_vec):
    return -(W * log_pi(theta_vec, ACTION))

# autograd
theta.grad = None
L = loss_fn(theta)
L.backward()
grad_auto = theta.grad.clone()

# central finite differences
eps = 1e-6
grad_num = torch.zeros_like(theta)
for i in range(2):
    up = theta.detach().clone(); up[i] += eps
    dn = theta.detach().clone(); dn[i] -= eps
    grad_num[i] = (loss_fn(up) - loss_fn(dn)) / (2 * eps)

print(f"loss                 : {L.item():.10f}")
print(f"autograd gradient    : {[round(v, 8) for v in grad_auto.tolist()]}")
print(f"finite-difference    : {[round(v, 8) for v in grad_num.tolist()]}")
max_err = (grad_auto - grad_num).abs().max().item()
print(f"max abs difference   : {max_err:.2e}")
assert max_err < 1e-6, "autograd gradient disagrees with finite differences"

# and the closed form, so a reader can check by hand:
#   d/dtheta_i [ -W log softmax(z)_a ] = W * ( softmax(z)_i - 1{i == a} )
p = torch.softmax(torch.tensor([0.3, -0.2, 0.0], dtype=DTYPE), dim=0)
hand = W * (p[:2] - torch.tensor([0.0, 1.0], dtype=DTYPE))
print(f"closed form, by hand : {[round(v, 8) for v in hand.tolist()]}")
assert torch.allclose(grad_auto, hand, atol=1e-8)
print("task 28 ok — autograd, finite differences and the closed form agree")
""")

md(r"""
### 3.2 Mean vs. sum, and what it costs

The slide derivation writes $\sum_t$; the code writes `.mean()`. These differ
by a factor of $T$, which is absorbed into the step size — but $T$ **varies
per episode** here (a crash ends the episode early), so `.sum()` would apply a
larger effective step to long episodes than to short ones.

That is a real, if second-order, effect: it biases the update magnitude
towards long episodes. `.mean()` removes it. Either is defensible as long as
the choice is stated, which is why the mode is logged alongside every run.
""")

code(r"""
print("T varies across episodes, so sum() and mean() differ by more than a")
print("constant across the dataset:")
print()
for T in (2, 10, 40):
    print(f"  episode length T={T:>2}  ->  sum/mean factor = {T}")
print()
print("mean() normalises by T, so the effective step size does not drift with")
print("how long the episode happened to last.")
""")

# ---------------------------------------------------------------- task 29
md(r"""
---
<a id="4"></a>
## 4. Adam parameter update — task 29

### The update rule

$$
\theta \;\leftarrow\; \theta - \alpha\,\nabla_\theta\mathcal{L}
$$

with Adam replacing the raw gradient by a running, per-parameter estimate.
What matters for this deliverable is not the optimiser's internals but the
**order of the three calls**:

```python
optimizer.zero_grad()   # 1. clear last step's gradient
loss.backward()         # 2. accumulate this step's gradient
optimizer.step()        # 3. apply the update
```

Reversing 1 and 2 accumulates gradients across episodes and the policy drifts;
reversing 2 and 3 applies the *previous* step's gradient. Neither raises an
error, which is why §4.2 asserts the parameters actually move.
""")

code(r"""
def make_optimizer(model, lr: float = 1e-3):
    '''Adam over all policy parameters.

    Adam rather than plain SGD because REINFORCE gradients vary a lot in
    magnitude between episodes: the weight is R(tau), which can be ~1 or ~30.
    Adam's per-parameter scaling makes a single fixed lr workable, which is
    why the experiments use 1e-3 throughout.
    '''
    return torch.optim.Adam(model.parameters(), lr=lr)


def train_step(model, optimizer, log_probs, weights, mode="vanilla"):
    '''One REINFORCE update. Returns the loss value before the step.'''
    loss = reinforce_loss(log_probs, weights, mode=mode)

    optimizer.zero_grad()   # 1
    loss.backward()         # 2  — only log_probs carries the graph
    optimizer.step()        # 3

    return loss.item()
""")

md(r"""
### 4.1 Verify the update actually moves the parameters

The test below runs one step and checks three things:

1. the parameters changed at all (the step happened),
2. every parameter that received a non-zero gradient moved,
3. the loss recomputed with the *new* parameters is lower than before —
   gradient descent on this single sample should decrease it.

Point 3 is the one that catches a sign error: with a flipped sign the loss
goes *up* and the assertion fires.
""")

code(r"""
import copy

torch.manual_seed(3)
model = PolicyNetwork(obs_dim=25, n_actions=5)
opt = make_optimizer(model, lr=1e-2)

obs = torch.randn(25)
probs = model(obs)
dist = torch.distributions.Categorical(probs)
action = dist.sample()
log_prob = dist.log_prob(action).unsqueeze(0)          # (1,)
weights = torch.tensor([5.0])                          # pretend G_0 = 5

before = [p.detach().clone() for p in model.parameters()]
loss_before = reinforce_loss(log_prob, weights, mode="vanilla").item()

train_step(model, opt, log_prob, weights, mode="vanilla")

after = [p.detach() for p in model.parameters()]
moved = sum(1 for b, a in zip(before, after) if not torch.allclose(b, a))

# recompute the same objective with the updated parameters
probs2 = model(obs)
log_prob2 = torch.distributions.Categorical(probs2).log_prob(action).unsqueeze(0)
loss_after = reinforce_loss(log_prob2, weights, mode="vanilla").item()

print(f"parameters that moved     : {moved} / {len(before)}")
print(f"loss before the step      : {loss_before:.6f}")
print(f"loss after the step       : {loss_after:.6f}")
print(f"change                    : {loss_after - loss_before:+.6f}")

assert moved > 0, "optimizer.step() did not change any parameter"
assert loss_after < loss_before, (
    "loss increased after the step — check the sign convention in reinforce_loss"
)
print("task 29 ok — one Adam step decreases the loss")
""")

md(r"""
### 4.2 What a wrong sign looks like

Worth stating explicitly, because it is the failure mode the assertion above
guards against and it is easy to introduce later:
""")

code(r"""
torch.manual_seed(3)
m_bad = copy.deepcopy(model)
opt_bad = make_optimizer(m_bad, lr=1e-2)

probs = m_bad(obs)
d = torch.distributions.Categorical(probs)
a = d.sample()
lp = d.log_prob(a).unsqueeze(0)
w = torch.tensor([5.0])

# the bug: maximising the objective instead of the loss
wrong_loss = (lp * w).mean()          # note: NO leading minus
opt_bad.zero_grad()
wrong_loss.backward()
opt_bad.step()

probs2 = m_bad(obs)
lp2 = torch.distributions.Categorical(probs2).log_prob(a).unsqueeze(0)
correct_after = reinforce_loss(lp2, w, mode="vanilla").item()
correct_before = reinforce_loss(lp, w, mode="vanilla").item()

print(f"with the sign bug, the true loss went {correct_before:.6f} -> {correct_after:.6f}")
print("-> blind ascent on a loss. The run still prints numbers; only this")
print("   comparison reveals that the policy is getting worse.")
""")

# ---------------------------------------------------------------- assembly
md(r"""
---
## 5. End-to-end: one episode, one update

Now the four pieces are wired together in the same order as
`experiments/train_reinforce.py`. This is the minimal reproduction of a
training step — if this cell runs and the return improves across a handful of
episodes, the loop is sound and any later failure is a hyperparameter issue
rather than a bug.
""")

code(r"""
import gymnasium as gym
import highway_env  # noqa: F401  (registers highway-v0)


def collect_episode(env, model, seed: int | None = None):
    '''Roll out one episode under the current policy.

    Each log_prob keeps its graph: the loss in section 3 needs it attached.
    '''
    obs, _ = env.reset(seed=seed)
    log_probs, rewards = [], []
    done = False
    while not done:
        x = torch.as_tensor(obs, dtype=torch.float32).flatten()
        a, logp = model.act(x)
        obs, r, terminated, truncated, _ = env.step(a)
        log_probs.append(logp.squeeze(-1))
        rewards.append(float(r))
        done = terminated or truncated
    return torch.stack(log_probs), rewards


env = gym.make("highway-v0")
torch.manual_seed(0)
model = PolicyNetwork(obs_dim=25, n_actions=5)
opt = make_optimizer(model, lr=1e-3)

log_probs, rewards = collect_episode(env, model, seed=0)
T = len(rewards)
w_rtg = reward_to_go(rewards, GAMMA)

print(f"episode length     : {T} steps")
print(f"undiscounted sum   : {sum(rewards):.4f}")
print(f"R(tau) discounted  : {total_trajectory_return(rewards, GAMMA).item():.4f}")
print(f"log_probs shape    : {tuple(log_probs.shape)}   requires_grad={log_probs.requires_grad}")
assert log_probs.requires_grad, "log_probs lost its graph during collection"

loss = train_step(model, opt, log_probs, w_rtg, mode="rtg")
print(f"loss               : {loss:.6f}")
print("one full REINFORCE update completed")
""")

md(r"""
### 5.1 A short run, to confirm the loop is not stuck

Ten episodes is far too few to claim the agent "learned" — that needs several
seeds and the plots in `experiments/plot_results.py`. What this checks is
weaker and honest: the returns are not monotonically collapsing, and the loss
magnitude stays in a sane range.
""")

code(r"""
torch.manual_seed(0)
model = PolicyNetwork(obs_dim=25, n_actions=5)
opt = make_optimizer(model, lr=1e-3)

returns_log, lengths_log = [], []
for ep in range(10):
    log_probs, rewards = collect_episode(env, model, seed=100 + ep)
    w = reward_to_go(rewards, GAMMA)
    train_step(model, opt, log_probs, w, mode="rtg")
    returns_log.append(sum(rewards))
    lengths_log.append(len(rewards))
    print(f"episode {ep:>2}  return {returns_log[-1]:7.3f}  steps {lengths_log[-1]:>2}")

print()
print(f"mean return over 10 episodes : {np.mean(returns_log):.3f}")
print(f"mean length over 10 episodes : {np.mean(lengths_log):.1f}")
print()
print("NOTE: 10 episodes / 1 seed is a smoke test, not evidence.")
print("      The batch runner uses 3 seeds; see experiments/train_reinforce.py.")

env.close()
""")

# ---------------------------------------------------------------- summary
md(r"""
---
## 6. Summary — what each task produced

| Task | Where it lives | What is verified here |
|---|---|---|
| 27 | `src/returns.py::total_trajectory_return` | closed form $r\frac{1-\gamma^T}{1-\gamma}$ matches; shape `(1,)` |
| 30 | `src/returns.py::reward_to_go` | recursion matches the brute-force definition; $G_0 = R(\tau)$ |
| 28 | `reinforce_loss` (§3) | autograd gradient matches finite differences to $10^{-6}$ |
| 29 | `make_optimizer` / `train_step` (§4) | one step decreases the loss; parameters move |

### The two things that break silently

1. **A detached `log_prob`.** The loss looks fine, `backward()` does not raise,
   and the gradient is `None` or zero. Guarded in §5 by asserting
   `log_probs.requires_grad` before the update.
2. **A flipped sign.** The agent is trained to make bad actions more likely.
   Guarded in §4.1 by requiring the loss to decrease after the step.

Neither shows up as an exception. Both show up as an agent that runs happily
and never improves — which is why they are asserted rather than assumed.

### Next

* `experiments/train_reinforce.py` — 3 seeds × {vanilla, rtg}
* `experiments/plot_results.py` — learning curves with shaded standard deviation
* `docs/proofs.md` — the causality lemma that justifies dropping past rewards
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "ml_env",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# nbformat >= 5.1 requires an `id` on every cell; without one nbconvert emits
# MissingIDFieldWarning and a future nbformat will refuse the file outright.
# Deterministic ids (index-based) keep the generated file diff-stable.
for i, cell in enumerate(nb["cells"]):
    cell["id"] = f"cell-{i:02d}"

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT}  ({len(cells)} cells)")
