"""Record videos of the trained agent (task 43) and the untrained one (task 42).

Trains a policy for N episodes, then records a deterministic (argmax) rollout
of the trained policy. Two clips are written:

    results/videos/trained_episode_{i}.gif      greedy policy after training
    results/videos/untrained_vs_trained.gif     side-by-side, same seed

Task 42 (untrained agent crashing) was already satisfied by the Week 1 smoke
test, but that clip and the trained one come from different setups, so the
side-by-side comparison regenerates both under identical conditions — which is
what a reader comparing "before vs after" actually needs.

Usage
-----
    python experiments/record_demo.py --epochs 300 --episodes 2

Note on --epochs
----------------
300, not 60. Measured on this setup, reward-to-go needs roughly 130 episodes
before the return leaves its initial plateau (8.19 -> 27.12 over 300 in
run_batch). A 60-episode clip shows a policy that has not started learning
yet, which would misrepresent the results table.

Why train here instead of loading a checkpoint
----------------------------------------------
There is no committed checkpoint (`*.pt` is gitignored, by design — it is
regenerable). Recording therefore trains the policy in-process, which also
keeps the clip honest: the video is produced by exactly the code in
`train_reinforce.py`, in the same run.

The trained agent is evaluated with `deterministic=True` (argmax). A stochastic
policy's clip varies between viewings and can show a bad sample even when the
policy is good; argmax shows what the policy actually prefers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import highway_env  # noqa: F401  (registers highway-v0)
import imageio
import numpy as np
import torch
from torch.optim import Adam

from src.policy import PolicyNetwork
from src.returns import reward_to_go

OUT_DIR = ROOT / "results" / "videos"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 8

# Must match experiments/train_reinforce.py ENV_CONFIG — see the comment there.
# `screen_width`/`screen_height` are render-only: they do not affect dynamics,
# only the size of the frames written to the GIF. The 150x600 default is too
# small to read when projected.
ENV_CONFIG = {"vehicles_count": 20}
RENDER_CONFIG = {"vehicles_count": 20,
                 "screen_width": 1200, "screen_height": 300}



def rollout(env, policy, seed: int, deterministic: bool,
            collect_frames: bool = True):
    """Run one episode and collect frames, return, steps and collisions.

    Pass collect_frames=False for evaluation. Calling env.render() consumes
    RNG, so an episode recorded with frames is not the same episode as one
    without them — measured: the same seed lasts 19 steps headless and 23 with
    rendering. Evaluation should use the unrendered dynamics.
    """
    obs, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    total_reward = 0.0
    steps = 0
    crashed = False
    done = False

    while not done:
        if collect_frames:
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))

        x = torch.as_tensor(obs, dtype=torch.float32).flatten()
        if policy is None:
            action = env.action_space.sample()          # random baseline
        else:
            with torch.no_grad():
                probs = policy(x)
                action = (int(torch.argmax(probs).item()) if deterministic
                          else int(torch.multinomial(probs, 1).item()))

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        # highway-env reports a crash through the reward being driven to 0 by
        # the collision term; `info["crashed"]` is present in 1.x
        if info.get("crashed"):
            crashed = True
        done = terminated or truncated

    if collect_frames:
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))
    return frames, total_reward, steps, crashed


def train(policy, env, epochs: int, seed: int, lr: float = 1e-3, gamma: float = 0.99):
    """REINFORCE with reward-to-go — the `rtg` variant from train_reinforce.py.

    Deliberately NO baseline here. The clip is meant to illustrate the row of
    the results table headed "Reward-to-go", so it must be produced by exactly
    that estimator; adding a running baseline would record a different
    algorithm from the one the numbers describe.
    """
    opt = Adam(policy.parameters(), lr=lr)
    torch.manual_seed(seed)
    history = []

    for ep in range(epochs):
        obs, _ = env.reset(seed=seed + ep)
        log_probs, rewards = [], []
        done = False
        while not done:
            x = torch.as_tensor(obs, dtype=torch.float32).flatten()
            a, logp = policy.act(x)
            obs, r, terminated, truncated, _ = env.step(a)
            log_probs.append(logp.squeeze(-1))
            rewards.append(float(r))
            done = terminated or truncated

        weights = reward_to_go(rewards, gamma)
        loss = -(torch.stack(log_probs) * weights.detach()).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        ep_return = sum(rewards)
        history.append(ep_return)

        if (ep + 1) % 25 == 0:
            print(f"  episode {ep:>3}  return {history[-1]:6.2f}  "
                  f"last25 mean {np.mean(history[-25:]):6.2f}")

    return history


def side_by_side(frames_a, frames_b, label_a="untrained", label_b="trained"):
    """Stack two frame sequences horizontally, padding the shorter one."""
    n = max(len(frames_a), len(frames_b))
    out = []
    for i in range(n):
        fa = frames_a[min(i, len(frames_a) - 1)]
        fb = frames_b[min(i, len(frames_b) - 1)]
        h = min(fa.shape[0], fb.shape[0])
        if fa.shape[1] != fb.shape[1]:
            w = min(fa.shape[1], fb.shape[1])
            fa, fb = fa[:, :w], fb[:, :w]
        out.append(np.concatenate([fa[:h], fb[:h]], axis=1))
    return out


def evaluate(policy, env, seeds, deterministic: bool = True):
    """Run several episodes and report the distribution, not one lucky draw.

    A single greedy rollout says almost nothing: the return swings by a factor
    of two on initial conditions alone. Reporting mean and range over several
    seeds is what makes "trained vs untrained" a claim you can defend.
    """
    rets, crashes, steps = [], [], []
    for s in seeds:
        _, ret, n, crash = rollout(env, policy, seed=s, deterministic=deterministic,
                                   collect_frames=False)
        rets.append(ret)
        crashes.append(crash)
        steps.append(n)
    return rets, crashes, steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--episodes", type=int, default=2,
                    help="how many trained clips to record")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-seed", type=int, default=1000)
    args = ap.parse_args()

    env = gym.make("highway-v0", render_mode="rgb_array", config=RENDER_CONFIG)
    # Training must NOT happen on the rendering env: with render_mode set, every
    # env.step() also draws a frame, which measured ~5x slower here. train()
    # never reads a frame, so it gets its own headless env.
    train_env = gym.make("highway-v0", config=ENV_CONFIG)
    policy = PolicyNetwork(obs_dim=25, n_actions=5)

    # ---------------------------------------------------- untrained clip
    print("recording UNTRAINED (random) policy...")
    un_frames, un_ret, un_steps, un_crash = rollout(
        env, None, seed=args.eval_seed, deterministic=False
    )
    print(f"  return {un_ret:.2f}  steps {un_steps}  crashed={un_crash}")

    # ---------------------------------------------------- train
    print(f"\ntraining for {args.epochs} episodes...")
    hist = train(policy, train_env, args.epochs, args.seed)
    train_env.close()
    print(f"  first 10 mean {np.mean(hist[:10]):.2f}  ->  "
          f"last 10 mean {np.mean(hist[-10:]):.2f}")

    # ---------------------------------------------------- trained clips
    print("\nrecording TRAINED (argmax) policy...")
    first_trained = None
    for i in range(args.episodes):
        frames, ret, steps, crash = rollout(
            env, policy, seed=args.eval_seed + i, deterministic=True
        )
        path = OUT_DIR / f"trained_episode_{i}.gif"
        imageio.mimsave(path, frames, fps=FPS)
        print(f"  {path.name}: return {ret:.2f}  steps {steps}  "
              f"crashed={crash}  ({len(frames)} frames)")
        if first_trained is None:
            first_trained = frames

    # ---------------------------------------------------- honest comparison
    # Evaluate on a HEADLESS env, not the rendering one.
    #
    # `env.render()` consumes RNG, so the same seed produces a different
    # episode depending on whether frames are being drawn: measured here,
    # seed 1000 on a random policy lasts 19 steps headless and 23 steps with
    # render_mode="rgb_array". Evaluating on the render env therefore measures
    # a task nobody trained on, and it reported crashes 10/10 for a policy that
    # crashes 0/10 when evaluated properly.
    eval_env = gym.make("highway-v0", config=ENV_CONFIG)
    eval_seeds = [args.eval_seed + i for i in range(10)]
    tr_rets, tr_crash, tr_steps = evaluate(policy, eval_env, eval_seeds,
                                           deterministic=True)
    un_rets, un_crash, un_steps = evaluate(None, eval_env, eval_seeds,
                                           deterministic=False)
    eval_env.close()
    print("\nevaluation over %d seeds (same seeds for both, headless env):"
          % len(eval_seeds))
    print("  untrained  mean %6.2f   range [%6.2f, %6.2f]   "
          "steps %4.1f   crashes %d/%d"
          % (np.mean(un_rets), min(un_rets), max(un_rets),
             np.mean(un_steps), sum(un_crash), len(un_crash)))
    print("  trained    mean %6.2f   range [%6.2f, %6.2f]   "
          "steps %4.1f   crashes %d/%d"
          % (np.mean(tr_rets), min(tr_rets), max(tr_rets),
             np.mean(tr_steps), sum(tr_crash), len(tr_crash)))

    # ---------------------------------------------------- side by side
    if first_trained:
        combo = side_by_side(un_frames, first_trained)
        path = OUT_DIR / "untrained_vs_trained.gif"
        imageio.mimsave(path, combo, fps=FPS)
        print(f"  {path.name}: {len(combo)} frames "
              f"(left = untrained, right = trained)")

    env.close()
    print(f"\nclips written to {OUT_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
