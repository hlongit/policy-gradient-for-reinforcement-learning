"""Week 2 — rollout test: verifies the policy network + action sampling.

Runs one episode of highway-v0 using PolicyNetwork.act, then prints the
shapes of the collected tensors:
    states      : (T, 5, 5)   raw obs matrix per step
    obs_flat    : (T, 25)     flattened for the network input
    actions     : (T,)        int64
    log_probs   : (T,)        float32
    rewards     : (T,)        float32

Also checks that log_probs are ≤ 0 (probabilities ≤ 1) and that the
flattened observation matches the policy's expected input dimension.

Run with:
    python tests/rollout_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.policy import PolicyNetwork

import gymnasium as gym
import highway_env  # noqa: F401  (registers highway-v0)
import torch

OBS_DIM = 25
N_ACTIONS = 5


def main() -> None:
    env = gym.make("highway-v0", render_mode="rgb_array")
    obs, info = env.reset(seed=0)

    model = PolicyNetwork(obs_dim=OBS_DIM, n_actions=N_ACTIONS)

    states: list[torch.Tensor] = []
    actions: list[int] = []
    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []

    done = False
    while not done:
        states.append(torch.from_numpy(obs).float())

        action, log_prob = model.act(torch.from_numpy(obs).flatten().float())
        actions.append(action)
        log_probs.append(log_prob)

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated

    env.close()

    # stack for shape reporting
    states_t = torch.stack(states)
    obs_flat = states_t.reshape(states_t.shape[0], -1)
    actions_t = torch.tensor(actions, dtype=torch.int64)
    log_probs_t = torch.stack(log_probs)
    rewards_t = torch.tensor(rewards, dtype=torch.float32)

    print("=== Rollout tensor shapes ===")
    print(f"states      : {tuple(states_t.shape)}   dtype {states_t.dtype}")
    print(f"obs_flat    : {tuple(obs_flat.shape)}  dtype {obs_flat.dtype}")
    print(f"actions     : {tuple(actions_t.shape)}  dtype {actions_t.dtype}")
    print(f"log_probs   : {tuple(log_probs_t.shape)}  dtype {log_probs_t.dtype}")
    print(f"rewards     : {tuple(rewards_t.shape)}  dtype {rewards_t.dtype}")

    # assertions
    assert obs_flat.shape[1] == OBS_DIM, "network input dim mismatch"
    assert actions_t.shape[0] == obs_flat.shape[0] == log_probs_t.shape[0] == rewards_t.shape[0]
    assert (log_probs_t <= 0).all(), "log-probs should be ≤ 0"
    print("rollout test passed")


if __name__ == "__main__":
    main()
