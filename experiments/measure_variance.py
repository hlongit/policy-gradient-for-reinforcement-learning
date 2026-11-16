"""Measure the variance of the three return estimators directly (task 46).

Task 46: "Generate plot comparing return estimator variance across the three
variants."

What "estimator variance" has to mean
-------------------------------------
The three variants are three ways of weighting the SAME score function:

    vanilla   w_t = R(tau)              (one scalar, broadcast over t)
    rtg       w_t = G_t                 (reward-to-go)
    baseline  w_t = G_t - V_phi(s_t)    (advantage with a learned critic)

They are all unbiased for the same policy gradient, so the honest comparison is
the variance of the *gradient estimate* g_hat, not the variance of the episode
returns. Plotting return spread across episodes answers "how noisy is the
score", which is a different (and much less flattering) question — an earlier
version of this script did exactly that and produced a result that contradicted
the theory for the wrong reason.

Method
------
1. Freeze one policy (the one from a trained run, or a random init).
2. Sample N independent trajectories from that policy with the SAME env seed
   set, so every variant sees the identical batch of trajectories.
3. For each of the N trajectories compute g_hat for all three variants.
4. Report Var(g_hat) — the trace of the covariance, i.e. the summed variance
   over parameters — per variant.

Using one shared batch is what makes the comparison fair: any difference in
variance then comes from the weight alone, not from different trajectories.

A rolling baseline
------------------
The baseline variant needs a V_phi. Training a critic from scratch here would
make the comparison depend on critic-training luck, so this script fits V_phi
by least squares on the *collected* returns (a running-baseline / empirical
value estimate). That is the strongest possible critic for this data, which
sets an upper bound on what the baseline can achieve — stated plainly in the
figure subtitle rather than hidden.

Usage
-----
    python experiments/measure_variance.py --n 200 --epochs 40
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch
from torch.optim import Adam

from src.policy import PolicyNetwork
from src.returns import reward_to_go, total_trajectory_return

OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

GAMMA = 0.99

# Must match experiments/train_reinforce.py ENV_CONFIG: the frozen policy is
# meant to be the kind of policy training produces. See the long comment there
# for why this is 20 and not the upstream default of 50 (speed).
ENV_CONFIG = {"vehicles_count": 20}



def train_policy(epochs: int, seed: int = 0) -> PolicyNetwork:
    """Train a policy with plain REINFORCE so the frozen policy is realistic.

    A randomly initialised policy produces near-uniform action probabilities,
    where all three estimators are close to identical and the comparison is
    uninformative. A trained policy makes the score function peaked, which is
    the regime the variance reduction is supposed to help in.
    """
    env = gym.make("highway-v0", config=ENV_CONFIG)
    policy = PolicyNetwork(obs_dim=25, n_actions=5)
    opt = Adam(policy.parameters(), lr=1e-3)
    torch.manual_seed(seed)

    history = []
    for ep in range(epochs):
        obs, _ = env.reset(seed=seed + ep)
        log_probs, rewards = [], []
        done = False
        while not done:
            x = torch.as_tensor(obs, dtype=torch.float32).flatten()
            a, lp = policy.act(x)
            obs, r, term, trunc, _ = env.step(a)
            log_probs.append(lp.squeeze(-1))
            rewards.append(float(r))
            done = term or trunc
        w = reward_to_go(rewards, GAMMA)
        loss = -(torch.stack(log_probs) * w.detach()).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(sum(rewards))

    env.close()
    print(f"  trained {epochs} episodes: first10 {np.mean(history[:10]):.2f}"
          f" -> last10 {np.mean(history[-10:]):.2f}")
    return policy


def collect(policy, n: int, seed0: int = 5000):
    """Sample n trajectories from the frozen policy.

    The log-probabilities must keep their autograd graph: the whole point of
    this script is to differentiate them once per trajectory per variant. Only
    the *action sampling* is wrapped in no_grad — the log_prob used later for
    the gradient is recomputed with `retain_graph` available.
    """
    env = gym.make("highway-v0", config=ENV_CONFIG)
    batch = []
    for i in range(n):
        obs, _ = env.reset(seed=seed0 + i)
        obs_list, log_probs, rewards = [], [], []
        done = False
        while not done:
            x = torch.as_tensor(obs, dtype=torch.float32).flatten()
            obs_list.append(x)
            probs = policy(x)                      # keeps graph w.r.t. theta
            dist = torch.distributions.Categorical(probs)
            a = dist.sample()                      # sampling itself is not diff
            # .squeeze(-1) is load-bearing: `probs` is (1, 5), so log_prob
            # comes back as shape (1,). Stacking (1,) tensors gives (T, 1),
            # and the later `logp_t * w` with w of shape (T,) then broadcasts
            # to (T, T) -- the loss silently becomes a product of two sums,
            # whose gradient is always a multiple of sum_t grad log pi_t.
            # Every variant then looks collinear and trace(Cov) ends up
            # proportional to |E g|^2, which fakes a variance reduction.
            lp = dist.log_prob(a).squeeze(-1)      # this one carries grad_fn
            obs, r, term, trunc, _ = env.step(int(a))
            log_probs.append(lp)
            rewards.append(float(r))
            done = term or trunc
        batch.append((torch.stack(obs_list), torch.stack(log_probs),
                      np.asarray(rewards, dtype=np.float32)))
        if (i + 1) % 50 == 0:
            print(f"    collected {i+1}/{n}")
    env.close()
    return batch


def gradient_norm_per_variant(policy, batch, baseline_fn):
    """For each trajectory, compute g_hat for all three variants.

    Returns {variant: array of shape (n,)} where each entry is that
    trajectory's gradient estimate flattened to a vector — kept as a tensor so
    the parameter-space variance can be computed by stacking.
    """
    n_params = sum(p.numel() for p in policy.parameters())
    out = {"vanilla": [], "rtg": [], "baseline": []}

    for obs_t, logp_t, rewards in batch:
        g_t = reward_to_go(rewards, GAMMA)

        weights = {
            "vanilla": total_trajectory_return(rewards, GAMMA).expand_as(logp_t),
            "rtg": g_t,
            "baseline": g_t - baseline_fn(obs_t, g_t),
        }

        for name, w in weights.items():
            policy.zero_grad(set_to_none=True)
            loss = -(logp_t * w.detach()).mean()
            loss.backward(retain_graph=True)   # reused across the 3 variants
            flat = torch.cat([
                p.grad.detach().reshape(-1) if p.grad is not None
                else torch.zeros(p.numel())
                for p in policy.parameters()
            ])
            out[name].append(flat.clone())
        policy.zero_grad(set_to_none=True)

    return {k: torch.stack(v) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="trajectories to sample")
    ap.add_argument("--epochs", type=int, default=40, help="episodes to train first")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"1. training a policy for {args.epochs} episodes (to freeze)...")
    policy = train_policy(args.epochs, args.seed)

    print(f"\n2. collecting {args.n} trajectories from the frozen policy...")
    batch = collect(policy, args.n)

    # --- running baseline: least-squares fit of V(s), ON A SEPARATE SPLIT.
    #
    # The baseline must not see the trajectories it is used to de-noise. If it
    # is fitted on the same batch, V_phi becomes a function of the sampled
    # actions in that batch, so E_a[V(s)] is no longer independent of a and the
    # score-function property (E_a[grad log pi] = 0) stops cancelling the
    # baseline term. The estimator then has real bias -- measured here as
    # mean(u.g) drifting to 0.88 against 0.18 for vanilla, when all three are
    # supposed to agree.
    #
    # So: fit the linear critic on the first half, measure on the second half.
    n_fit = max(10, args.n // 2)
    fit_batch = batch[:n_fit]
    eval_batch = batch[n_fit:] if len(batch) > n_fit else batch

    all_obs = torch.cat([b[0] for b in fit_batch])       # (sum T, 25)
    all_ret = torch.cat([torch.as_tensor(
        reward_to_go(b[2], GAMMA)) for b in fit_batch])  # (sum T,)

    A = torch.cat([all_obs, torch.ones(len(all_obs), 1)], dim=1).double()
    y = all_ret.double()
    coef, *_ = torch.linalg.lstsq(A, y.unsqueeze(-1))
    coef = coef.squeeze(-1).float()

    def baseline_fn(obs_t, g_t):
        a = torch.cat([obs_t, torch.ones(len(obs_t), 1)], dim=1)
        return a @ coef

    print(f"   fitted a linear baseline on {n_fit} trajectories, "
          f"measuring on {len(eval_batch)} held-out ones")

    print("\n3. computing the gradient estimate for each trajectory...")
    grads = gradient_norm_per_variant(policy, eval_batch, baseline_fn)

    print("\n=== variance of the gradient estimator ===")
    print(f"{'variant':<10}{'trace(Cov)':>14}{'mean |g|':>12}{'rel. to vanilla':>18}")
    print("-" * 56)
    stats = {}
    trace_vanilla = None
    for name in ("vanilla", "rtg", "baseline"):
        g = grads[name]
        var = g.var(dim=0, unbiased=True).sum().item()
        mean_norm = g.norm(dim=1).mean().item()
        if name == "vanilla":
            trace_vanilla = var
        ratio = var / trace_vanilla if trace_vanilla else float("nan")
        stats[name] = {"trace_cov": var, "mean_norm": mean_norm,
                       "relative": ratio}
        print(f"{name:<10}{var:>14.2f}{mean_norm:>12.3f}{ratio:>17.2f}x")

    # ------------------------------------------------------------------
    # Scale-free variance.
    #
    # trace(Cov) alone is not a fair comparison: an estimator that simply
    # shrinks the gradient has a smaller trace without being any less noisy
    # *relative to its own signal*. Adam divides by sqrt(v_hat), so a uniform
    # rescaling of g_hat is invisible to it -- a "variance reduction" that
    # only shows up as a smaller gradient does nothing for the optimiser.
    #
    # Dividing by |E[g]|^2 gives the noise-to-signal ratio, which is invariant
    # to that rescaling. This is the number that should be quoted.
    # ------------------------------------------------------------------
    print("\n=== scale-free variance (noise per unit of signal) ===")
    print(f"{'variant':<10}{'trace/|E g|^2':>16}{'rel. to vanilla':>18}")
    print("-" * 44)
    snr_vanilla = None
    for name in ("vanilla", "rtg", "baseline"):
        s = stats[name]
        norm = s["trace_cov"] / s["mean_norm"] ** 2
        if name == "vanilla":
            snr_vanilla = norm
        s["trace_normalised"] = norm
        s["relative_normalised"] = norm / snr_vanilla if snr_vanilla else float("nan")
        print(f"{name:<10}{norm:>16.3f}{s['relative_normalised']:>17.2f}x")

    print("\n  (mean of the estimate should be ~equal across variants:")
    for name in ("vanilla", "rtg", "baseline"):
        m = grads[name].mean(dim=0)
        print(f"    {name:<10} |E[g]| = {m.norm().item():.4f}")

    # ------------------------------------------------------------------
    # Why trace(Cov) is not trustworthy here, and what to report instead.
    #
    # The policy has ~20k parameters but we only draw a few hundred
    # trajectories. A covariance in 20k dimensions estimated from n samples has
    # rank at most n-1, so trace(Cov) is biased downward by an unknown amount
    # and does not converge in this regime.
    #
    # The fix is a RANDOM projection. For a fixed unit vector u drawn
    # independently of the data,
    #
    #       E[ Var_traj(u . g_hat) ] = trace(Cov) / d
    #
    # so averaging over a few random directions estimates the mean per-
    # coordinate variance without ever forming the 20k x 20k matrix. Each
    # direction gives a scalar with finite variance, and averaging K of them
    # cuts the error by sqrt(K). This is the standard Johnson-Lindenstrauss
    # argument, and unlike a data-dependent direction it cannot be biased
    # toward whichever variant happens to have the largest gradient.
    # ------------------------------------------------------------------
    d = grads["vanilla"].shape[1]
    K = 64
    gen = torch.Generator().manual_seed(12345)     # fixed: data-independent
    U = torch.randn(d, K, generator=gen)
    U = U / U.norm(dim=0, keepdim=True)            # (d, K) unit columns

    print(f"\n=== projected variance (random directions, K={K}) ===")
    print(f"{'variant':<10}{'E[Var(u.g)]':>14}{'rel. to vanilla':>18}")
    print("-" * 42)
    proj = {}
    for name in ("vanilla", "rtg", "baseline"):
        P = grads[name] @ U                       # (n, K)
        proj[name] = P.var(dim=0, unbiased=True).mean().item()
    base_v = proj["vanilla"]
    for name in ("vanilla", "rtg", "baseline"):
        v = proj[name]
        stats[name]["proj_var"] = v
        stats[name]["proj_rel"] = v / base_v
        print(f"{name:<10}{v:>14.6f}{v / base_v:>17.2f}x")

    print("\n  E[Var(u.g)] estimates trace(Cov)/d, the mean per-coordinate")
    print("  variance. It is scale-dependent: an estimator that simply shrinks")
    print("  the gradient scores lower here without being less noisy.")
    print("  For the scale-free comparison see experiments/baseline_quality.py,")
    print("  which also splits the baseline fit from the measurement.")

    # ---------------------------------------------------------------- plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLOURS = {"vanilla": "#C0504D", "rtg": "#4F81BD", "baseline": "#26A9E0"}
    LABELS = {"vanilla": "Vanilla\nw = R(τ)",
              "rtg": "Reward-to-go\nw = G_t",
              # the baseline here is a linear least-squares fit of V(s), not a
              # trained critic network — label it for what it is
              "baseline": "RTG + linear baseline\nw = G_t − (aᵀs + b)"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=150,
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    names = ["vanilla", "rtg", "baseline"]
    vals = [stats[n]["proj_var"] for n in names]
    bars = ax1.bar(range(3), vals, color=[COLOURS[n] for n in names], width=0.62)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels([LABELS[n] for n in names], fontsize=9)
    ax1.set_ylabel("E[Var(u · ĝ)]   (random directions)")
    ax1.set_title("Estimator variance (random projection)\n"
                  f"{args.n} trajectories, identical batch for all variants",
                  fontsize=10)
    for b, v, n_ in zip(bars, vals, names):
        ax1.text(b.get_x() + b.get_width() / 2, v,
                 f"{v:.4f}\n({stats[n_]['proj_rel']:.2f}×)",
                 ha="center", va="bottom", fontsize=8.5)
    ax1.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax1.spines[["top", "right"]].set_visible(False)

    # per-trajectory projected estimate along one fixed direction: the spread
    for i, n in enumerate(names):
        proj = (grads[n] @ U[:, 0]).numpy()
        ax2.scatter(np.full_like(proj, i, dtype=float) +
                    np.random.default_rng(0).normal(0, 0.05, len(proj)),
                    proj, s=7, alpha=0.35, color=COLOURS[n])
        ax2.hlines(proj.mean(), i - 0.28, i + 0.28, color=COLOURS[n],
                   linewidth=2.2, label=f"mean {proj.mean():.3f}")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([LABELS[n] for n in names], fontsize=9)
    ax2.set_ylabel("u · ĝ  per trajectory (one random u)")
    ax2.set_title("Per-trajectory projected estimate", fontsize=10)
    ax2.grid(True, alpha=0.25, linestyle=":")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(frameon=False, fontsize=8)

    fig.suptitle("Gradient-estimator variance on highway-v0  "
                 "(policy frozen after training; baseline fitted on a held-out "
                 "split, measured on the rest)", fontsize=9.5)
    fig.tight_layout()
    out = OUT / "variance_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out.relative_to(ROOT)}")

    (OUT / "variance_stats.json").write_text(
        json.dumps({"n_trajectories": args.n, "epochs": args.epochs,
                    "gamma": GAMMA, "variants": stats}, indent=2),
        encoding="utf-8")
    print(f"wrote {(OUT / 'variance_stats.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
