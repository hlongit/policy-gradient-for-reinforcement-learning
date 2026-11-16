"""Sanity-check the three estimators on a problem with a known answer.

Claim being tested: `vanilla`, `rtg` and `baseline` all estimate the SAME
policy gradient (unbiased), so E[g_hat] must agree across variants, while only
their variances should differ.

This runs on a 3-state, 2-action bandit-with-episode-length problem where the
true gradient can be computed by enumeration. It does not touch the highway
env, so it is fast and independent of the bug we just fixed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

GAMMA = 0.99
T = 4
N_TRAJ = 4000


def logits(theta: torch.Tensor) -> torch.Tensor:
    """2x2 parameterisation: logits[s, a] = theta[s, a]."""
    return theta


def policy_probs(theta):
    return torch.softmax(logits(theta), dim=-1)


def sample_episode(theta, gen):
    """State is fixed (contextual bandit repeated T times); reward ~ Bernoulli."""
    probs = policy_probs(theta)
    states = torch.arange(2).repeat_interleave(T // 2)
    acts = torch.stack([torch.multinomial(probs[s], 1, generator=gen).squeeze()
                        for s in states])
    rew = torch.bernoulli(torch.tensor([1.0 if a == s else 0.0
                                        for s, a in zip(states, acts)]))
    return states, acts, rew


def true_grad(theta):
    """Exact gradient of J = E[sum_t gamma^t r_t].

    Verified against finite differences in tests/grad_check.py (agreement to
    ~1e-10 in float64). Note there is NO extra factor of pi(a|s) inside the
    sum: d/dtheta log pi(s|s) already carries it, because
    d pi(s|s)/dtheta[s,:] = pi(s|s) * (onehot(s) - pi[s,:]).
    """
    probs = policy_probs(theta)
    g = torch.zeros_like(theta)
    eye = torch.eye(2, dtype=probs.dtype)
    for t in range(T):
        s = t % 2
        g[s, :] += (GAMMA ** t) * probs[s, s] * (eye[s] - probs[s, :])
    return g


def estimate(theta, variant, n=N_TRAJ, seed=0):
    gen = torch.Generator().manual_seed(seed)
    grads = []
    baseline_running = 0.0
    for i in range(n):
        states, acts, rew = sample_episode(theta, gen)
        probs = policy_probs(theta)
        logp = torch.stack([torch.log(probs[s, a]) for s, a in zip(states, acts)])

        # reward-to-go and full return
        rtg = torch.zeros(T)
        acc = 0.0
        rl = rew.tolist()
        for t in reversed(range(T)):
            acc = rl[t] + GAMMA * acc
            rtg[t] = acc
        full = sum(GAMMA ** t * rl[t] for t in range(T))

        if variant == "vanilla":
            w = torch.full((T,), full)
        elif variant == "rtg":
            w = rtg
        else:
            w = rtg - baseline_running
            baseline_running = 0.9 * baseline_running + 0.1 * full

        theta.grad = None
        loss = -(logp * w.detach()).mean()
        loss.backward()
        # Two corrections, both of them conventions rather than bugs:
        #  * `backward()` gives the gradient of the LOSS, i.e. -grad J;
        #    the policy gradient (ascent direction) is its negation.
        #  * the loss uses .mean() over the T steps, which divides the
        #    gradient by T. The estimator of grad J is the SUM, so scale back.
        grads.append(-theta.grad.clone() * T)
        theta.grad = None
    return torch.stack(grads)


def main():
    torch.manual_seed(0)
    theta0 = torch.zeros(2, 2, requires_grad=True)
    gt = true_grad(theta0.detach()).flatten()

    print("true gradient (enumerated):", " ".join("%+.4f" % v for v in gt.tolist()))
    print()
    print("%-10s %12s %12s %12s" % ("variant", "|E g - g*|", "cos(E g, g*)", "trace(Cov)"))
    print("-" * 50)
    for v in ("vanilla", "rtg", "baseline"):
        th = theta0.detach().clone().requires_grad_(True)
        G = estimate(th, v)
        eg = G.mean(dim=0).flatten()
        err = (eg - gt).norm().item()
        cos = torch.nn.functional.cosine_similarity(eg, gt, dim=0).item()
        var = G.var(dim=0, unbiased=True).sum().item()
        print("%-10s %12.4f %12.4f %12.2f" % (v, err, cos, var))

    print()
    print("Expectation: |E g - g*| small for all three (unbiased),")
    print("cos ~ 1 for all three, and trace(Cov) falling vanilla > rtg > baseline.")


if __name__ == "__main__":
    main()
