"""REINFORCE training loop — vanilla, reward-to-go, and learned-baseline.

Variants
--------
vanilla   weight = R(tau)                 the whole-episode return, broadcast
rtg       weight = G_t                    reward-to-go, per step
baseline  weight = G_t - V_phi(s_t)       reward-to-go minus a learned critic value

All three are unbiased estimators of the same policy gradient; they differ only
in variance. `baseline` adds a critic trained by regression onto the sampled
returns, which is why it needs a second optimiser.

Architecture matches src/policy.py (25 -> 128 -> 128 -> 5 softmax) and
src/critic.py (25 -> 128 -> 128 -> 1).

Run with:
    python experiments/train_reinforce.py --mode rtg --seed 0 --epochs 30

Outputs:
    results/logs/reinforce_{seed}_{mode}.csv
        columns: episode, total_return, steps, loss, grad_norm, critic_loss

`grad_norm` is the L2 norm of the policy gradient before the optimiser step
(task 41). It is logged for every variant so the plots can show how the
gradient scale evolves, not just the return.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import highway_env  # noqa: F401  (registers highway-v0)
import torch
from torch.optim import Adam

from src.critic import ValueNetwork, critic_loss
from src.policy import PolicyNetwork
from src.returns import reward_to_go, total_trajectory_return

LOG_DIR = Path("results/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0
EPOCHS = 30
GAMMA = 0.99
LR = 1e-3
CRITIC_LR = 1e-2        # the critic regresses onto a bounded target; it can
                        # take a larger step than the policy without oscillating

MODES = ("vanilla", "rtg", "baseline")

# ---------------------------------------------------------------------------
# Environment configuration — read this before comparing to another run.
#
# highway-v0 ships with `vehicles_count: 50`. On the machine used for these
# experiments that costs ~680 ms per env.step() (measured), so one 15-step
# episode takes ~10 s and a REINFORCE run of a few thousand episodes takes
# hours. REINFORCE needs thousands of episodes to move: with 40 updates the
# policy flatlines (measured: first-10 mean 13.13 -> last-10 mean 13.05).
#
# `vehicles_count: 20` measures ~155 ms/step here — 4.4x faster, which is what
# makes a real training run fit in the time budget. It is a *different, easier*
# task (less traffic), so numbers from it are NOT comparable to logs produced
# with 50 vehicles. Every results table must state which config produced it.
# ---------------------------------------------------------------------------
ENV_CONFIG = {"vehicles_count": 20}


CSV_HEADER = ["episode", "total_return", "steps", "loss", "grad_norm",
              "critic_loss"]


def collect_trajectory(env, model, seed: int):
    """Run one episode under the current policy.

    Returns (obs_list, actions, log_probs, rewards). Every log_prob keeps its
    autograd graph — the loss is defined by its gradient.
    """
    obs, info = env.reset(seed=seed)
    log_probs: list[torch.Tensor] = []
    actions: list[int] = []
    obs_list: list[torch.Tensor] = []
    rewards: list[float] = []
    done = False
    while not done:
        obs_flat = torch.from_numpy(obs).flatten().float()
        obs_list.append(obs_flat)
        action, log_prob = model.act(obs_flat)
        actions.append(action)
        log_probs.append(log_prob.squeeze(-1))  # (T,) after stack
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
    return obs_list, actions, log_probs, rewards


def reinforce_loss(log_probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Policy-gradient loss:  L = -mean( log_prob_t * weight_t ).

    Minimising this maximises E[weight * log pi(a|s)]. The weights are
    detached inside the callers — they are computed from recorded rewards and
    carry no theta.
    """
    return -(log_probs * weights).mean()


def grad_norm(module: torch.nn.Module) -> float:
    """L2 norm over all parameter gradients (task 41).

    Must be called AFTER backward() and BEFORE step(), otherwise the
    gradients have already been applied and zero_grad has cleared them.
    """
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total ** 0.5


def train_one_mode(mode: str, seed: int, epochs: int, gamma: float, lr: float,
                   critic_lr: float = CRITIC_LR):
    """Train one REINFORCE variant and return per-episode returns."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    torch.manual_seed(seed)

    env = gym.make("highway-v0", config=ENV_CONFIG)
    # No render_mode: training never reads a frame. Verified bit-identical:
    # with render_mode="rgb_array" and without it, episode 0 gives the same
    # return (15.715115222733763), the same step count and the same gradient
    # norm, because rendering does not touch the RNG that samples actions.
    # Dropping it removes the pygame surface allocation, which dominated the
    # per-episode cost. Clips are recorded separately by record_demo.py.
    model = PolicyNetwork(obs_dim=25, n_actions=5)
    optimizer = Adam(model.parameters(), lr=lr)

    critic = None
    critic_opt = None
    if mode == "baseline":
        critic = ValueNetwork(obs_dim=25)
        critic_opt = Adam(critic.parameters(), lr=critic_lr)

    # Re-seed AFTER constructing the networks.
    #
    # `baseline` builds a second network, and its weight init draws from the
    # global RNG. Without this re-seed the three variants would start from the
    # same policy but consume different random streams for action sampling, so
    # episode 0 would already differ — and the comparison across variants would
    # mix "different estimator" with "different sampling noise". Re-seeding
    # here gives every variant the same initial policy and the same action
    # stream, so the only difference left is the weight in the loss.
    torch.manual_seed(seed)

    returns_log: list[float] = []

    csv_path = LOG_DIR / f"reinforce_{seed}_{mode}.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)

    for epoch in range(epochs):
        obs_list, actions, log_probs, rewards = collect_trajectory(
            env, model, seed + epoch
        )
        log_probs_t = torch.stack(log_probs)          # (T,)
        g_t = reward_to_go(rewards, gamma)            # (T,)

        c_loss_value = float("nan")

        # --- choose the weight ------------------------------------------
        if mode == "vanilla":
            # one scalar for the whole episode, broadcast over the T steps
            weights = total_trajectory_return(rewards, gamma).expand_as(log_probs_t)

        elif mode == "rtg":
            weights = g_t

        else:  # baseline
            obs_batch = torch.stack(obs_list)          # (T, 25)
            values = critic(obs_batch).squeeze(-1)     # (T,) with graph in phi
            c_loss = critic_loss(values, g_t)

            # Baseline subtraction. `values.detach()` is essential: the weight
            # must carry no gradient into phi, or the policy update would
            # also try to move the critic toward whatever lowers the policy
            # loss — which is not the critic's objective.
            weights = g_t - values.detach()

            critic_opt.zero_grad()
            c_loss.backward()
            critic_opt.step()
            c_loss_value = float(c_loss.item())

        loss = reinforce_loss(log_probs_t, weights.detach())

        optimizer.zero_grad()
        loss.backward()
        g_norm = grad_norm(model)      # BEFORE step: grads still present
        optimizer.step()

        episode_return = float(sum(rewards))
        returns_log.append(episode_return)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, episode_return, len(rewards),
                round(float(loss.item()), 6), round(g_norm, 6),
                round(c_loss_value, 6) if mode == "baseline" else "",
            ])

        if (epoch + 1) % 5 == 0 or epoch == 0:
            extra = f"  critic {c_loss_value:7.4f}" if mode == "baseline" else ""
            print(
                f"[{mode} seed={seed}] epoch {epoch:3d}  "
                f"return {episode_return:6.2f}  steps {len(rewards):2d}  "
                f"loss {loss.item():8.4f}  |g| {g_norm:7.4f}{extra}"
            )

    env.close()
    return returns_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(MODES), default="rtg")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--critic-lr", type=float, default=CRITIC_LR)
    args = parser.parse_args()

    print(f"Training REINFORCE ({args.mode}) seed={args.seed} "
          f"epochs={args.epochs} gamma={args.gamma} lr={args.lr}")
    train_one_mode(args.mode, args.seed, args.epochs, args.gamma, args.lr,
                   args.critic_lr)
    print(f"Done. CSV log: results/logs/reinforce_{args.seed}_{args.mode}.csv")


if __name__ == "__main__":
    main()
