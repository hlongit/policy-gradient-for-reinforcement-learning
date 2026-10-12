"""Week 3 — REINFORCE training loop (vanilla vs reward-to-go).

Implements:
    - REINFORCE policy-gradient loss for both return estimators
    - Adam optimizer update step
    - One test run comparing vanilla return vs reward-to-go
      (logs episode returns to results/logs/)

Architecture matches src/policy.py (25 -> 128 -> 128 -> 5 softmax).

Run with:
    python experiments/train_reinforce.py [--mode {vanilla,rtg}] [--seed N] [--epochs N]

Outputs:
    results/logs/reinforce_{seed}_{mode}.csv  - episode, total_return, steps
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

from src.policy import PolicyNetwork
from src.returns import reward_to_go, total_trajectory_return

LOG_DIR = Path("results/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0
EPOCHS = 30
GAMMA = 0.99
LR = 1e-3


def collect_trajectory(
    env, model, seed: int
) -> tuple[list[torch.Tensor], list[int], list[torch.Tensor], list[float]]:
    """Run one episode under the current policy; return lists of tensors/scalars."""
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


def reinforce_loss(
    log_probs: torch.Tensor, returns: torch.Tensor
) -> torch.Tensor:
    """Policy-gradient loss:  L = -mean( log_prob_t * return_t ).

    Minimising this maximises E[G * log pi(a|s)].
    """
    return -(log_probs * returns).mean()


def train_one_mode(mode: str, seed: int, epochs: int, gamma: float, lr: float):
    """Train one REINFORCE variant and return per-episode returns."""
    torch.manual_seed(seed)

    env = gym.make("highway-v0", render_mode="rgb_array")
    model = PolicyNetwork(obs_dim=25, n_actions=5)
    optimizer = Adam(model.parameters(), lr=lr)

    returns_log: list[float] = []

    csv_path = LOG_DIR / f"reinforce_{seed}_{mode}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_return", "steps"])

    for epoch in range(epochs):
        obs_list, actions, log_probs, rewards = collect_trajectory(env, model, seed + epoch)

        # --- choose return estimator -----------------------------------
        if mode == "vanilla":
            # scalar return broadcast to every step
            G = total_trajectory_return(rewards, gamma)  # (1,)
            returns = G.expand(len(log_probs))            # (T,)
        elif mode == "rtg":
            returns = reward_to_go(rewards, gamma)        # (T,)
        else:
            raise ValueError(f"unknown mode {mode}")

        log_probs_t = torch.stack(log_probs)  # (T,)
        loss = reinforce_loss(log_probs_t, returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_return = float(sum(rewards))
        returns_log.append(episode_return)

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, episode_return, len(rewards)])

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"[{mode} seed={seed}] epoch {epoch:3d}  "
                f"return {episode_return:6.2f}  steps {len(rewards)}  "
                f"loss {loss.item():.4f}"
            )

    env.close()
    return returns_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vanilla", "rtg"], default="vanilla")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    print(f"Training REINFORCE ({args.mode}) seed={args.seed} "
          f"epochs={args.epochs} gamma={args.gamma} lr={args.lr}")
    train_one_mode(args.mode, args.seed, args.epochs, args.gamma, args.lr)
    print("Done. CSV logs in results/logs/")


if __name__ == "__main__":
    main()
