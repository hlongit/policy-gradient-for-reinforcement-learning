"""Verify the true policy gradient by finite differences, correctly.

Setup (deliberately tiny so everything is checkable by hand):
    T = 4 steps, state at step t is s_t = t % 2  (deterministic, so no
    transition dynamics to enumerate).
    Reward at step t is 1 if a_t == s_t, else 0.
    theta is a 2x2 table: logits[s, a].  The SAME theta[s, :] is used every
    time state s is visited, so the gradient accumulates over visits.

    J(theta) = sum_t gamma^t * P(a_t == s_t)
             = sum_t gamma^t * softmax(theta[s_t])[s_t]

Finite differences must match the analytic gradient of that expression.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

GAMMA = 0.99
T = 4


def J(theta):
    probs = torch.softmax(theta, dim=-1)
    total = torch.zeros((), dtype=probs.dtype)
    for t in range(T):
        s = t % 2
        total = total + (GAMMA ** t) * probs[s, s]
    return total


def analytic(theta):
    probs = torch.softmax(theta, dim=-1)
    g = torch.zeros_like(theta)
    for t in range(T):
        s = t % 2
        # d/dtheta[s,a] of probs[s,s] = probs[s,s]*(1[a==s] - probs[s,a])
        g[s, :] += (GAMMA ** t) * probs[s, s] * (
            torch.eye(2)[s] - probs[s, :])
    return g


def numeric(theta, eps=1e-5):
    g = torch.zeros_like(theta)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            tp, tm = theta.clone(), theta.clone()
            tp[i, j] += eps
            tm[i, j] -= eps
            g[i, j] = (J(tp) - J(tm)) / (2 * eps)
    return g


for label, th in [("theta = 0", torch.zeros(2, 2)),
                  ("asymmetric", torch.tensor([[0.3, -0.2], [-0.5, 0.8]]))]:
    a, n = analytic(th), numeric(th)
    print(label)
    print("  analytic:", [round(v, 6) for v in a.flatten().tolist()])
    print("  numeric :", [round(v, 6) for v in n.flatten().tolist()])
    print("  max |diff| = %.3e" % (a - n).abs().max().item())
    print("  J = %.6f" % J(th).item())
    print()
