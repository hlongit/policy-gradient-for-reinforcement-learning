"""Compare baseline QUALITY, not just baseline presence.

Theory says any b(s) independent of the action reduces variance, with the
reduction bounded by how well b predicts the return. A bad b can still add
noise. This measures three baselines on the same held-out trajectories:

  none       w = G_t                        (plain reward-to-go)
  constant   w = G_t - mean(G_t)            (one scalar, fitted on the fit split)
  linear     w = G_t - (A@coef + b)         (26-parameter linear critic)

Reported as Var(u.g) ratios, which are well defined and do not divide by a
near-zero signal.
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import gymnasium as gym
import highway_env  # noqa: F401

from src.policy import PolicyNetwork
from src.returns import reward_to_go, total_trajectory_return

GAMMA = 0.99
ENV_CONFIG = {"vehicles_count": 20}


def rollout(policy, env, seed):
    obs, _ = env.reset(seed=seed)
    obs_l, acts, rw = [], [], []
    done = False
    while not done:
        x = torch.as_tensor(obs, dtype=torch.float32).flatten()
        obs_l.append(x)
        with torch.no_grad():
            p = policy(x)
        a = int(torch.distributions.Categorical(p).sample())
        acts.append(a)
        obs, r, te, tr, _ = env.step(a)
        rw.append(float(r))
        done = te or tr
    return torch.stack(obs_l), acts, np.asarray(rw, np.float32)


def flat_grad(policy, obs_t, acts, weights):
    policy.zero_grad(set_to_none=True)
    probs = policy(obs_t)
    lp = torch.distributions.Categorical(probs).log_prob(
        torch.as_tensor(acts, dtype=torch.long))
    (-(lp * weights.detach()).mean()).backward()
    v = torch.cat([p.grad.reshape(-1) for p in policy.parameters()
                   if p.grad is not None]).clone()
    policy.zero_grad(set_to_none=True)
    return v


def main():
    n_fit, n_eval = 250, 250
    torch.manual_seed(0)
    policy = PolicyNetwork(25, 5)
    env = gym.make("highway-v0", config=ENV_CONFIG)

    print("collecting %d trajectories..." % (n_fit + n_eval))
    trajs = [rollout(policy, env, 9000 + i) for i in range(n_fit + n_eval)]
    env.close()
    fit, ev = trajs[:n_fit], trajs[n_fit:]

    # --- constant baseline: mean reward-to-go over the fit split
    c = torch.cat([reward_to_go(t[2], GAMMA) for t in fit]).mean()

    # --- linear critic fitted on the fit split
    A = torch.cat([torch.cat([t[0], torch.ones(len(t[0]), 1)], dim=1)
                   for t in fit]).double()
    y = torch.cat([reward_to_go(t[2], GAMMA) for t in fit]).double()
    coef, *_ = torch.linalg.lstsq(A, y.unsqueeze(-1))
    coef = coef.squeeze(-1).float()

    def lin(obs_t):
        return torch.cat([obs_t, torch.ones(len(obs_t), 1)], dim=1) @ coef

    # --- how well does each candidate predict G_t on the eval split?
    all_ev_obs = torch.cat([t[0] for t in ev])
    all_ev_g = torch.cat([reward_to_go(t[2], GAMMA) for t in ev])
    var_g = all_ev_g.var().item()
    pred_c = torch.full_like(all_ev_g, c)
    pred_l = lin(all_ev_obs)
    print()
    print("critic quality on the held-out split (variance of G_t explained)")
    print("  Var(G_t)                 = %.3f" % var_g)
    print("  constant baseline  R^2   = %.4f"
          % (1 - (all_ev_g - pred_c).var().item() / var_g))
    print("  linear critic      R^2   = %.4f"
          % (1 - (all_ev_g - pred_l).var().item() / var_g))

    variants = {
        "rtg": lambda o, g: g,
        "rtg-const": lambda o, g: g - c,
        "rtg-linear": lambda o, g: g - lin(o),
        "vanilla": lambda o, g: total_trajectory_return(
            torch.tensor([]), GAMMA) if False else None,   # handled below
    }

    print("\ncomputing gradients on the held-out split...")
    grads = {}
    for t in ev:
        obs_t, acts, rw = t
        g_t = reward_to_go(rw, GAMMA)
        full = total_trajectory_return(rw, GAMMA).expand_as(g_t)
        w = {
            "vanilla": full,
            "rtg": g_t,
            "rtg-const": g_t - c,
            "rtg-linear": g_t - lin(obs_t),
        }
        for k, ww in w.items():
            grads.setdefault(k, []).append(flat_grad(policy, obs_t, acts, ww))
    grads = {k: torch.stack(v) for k, v in grads.items()}

    d = grads["vanilla"].shape[1]
    gen = torch.Generator().manual_seed(7)
    U = torch.randn(d, 64, generator=gen)
    U = U / U.norm(dim=0, keepdim=True)

    base = None
    print()
    print("%-12s %14s %14s %10s" % ("variant", "E[Var(u.g)]", "rel. to vanilla",
                                    "gain"))
    print("-" * 54)
    rows = {}
    for k in ("vanilla", "rtg", "rtg-const", "rtg-linear"):
        P = grads[k] @ U
        v = P.var(dim=0, unbiased=True).mean().item()
        if base is None:
            base = v
        rows[k] = v
        print("%-12s %14.6f %13.2fx %9.2fx" % (k, v, v / base, base / v))
    print()
    print("gain = factor fewer trajectories than vanilla for equal accuracy.")

    out = ROOT / "results" / "figures" / "baseline_quality.json"
    out.write_text(json.dumps({
        "n_fit": n_fit, "n_eval": n_eval,
        "var_G_eval": var_g,
        "r2_constant": 1 - (all_ev_g - pred_c).var().item() / var_g,
        "r2_linear": 1 - (all_ev_g - pred_l).var().item() / var_g,
        "proj_var": rows,
    }, indent=2), encoding="utf-8")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
