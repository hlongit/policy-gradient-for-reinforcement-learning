"""Is the trained policy crashing early, or surviving to the 40-step limit?

`crashed` being True on 10/10 episodes is only alarming if the crash happens
early. highway-v0 also ends at duration=40, and a collision on the final step
is nearly irrelevant to the return. This separates the two cases.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch

from src.policy import PolicyNetwork

ENV_CONFIG = {"vehicles_count": 20}


def rollout_detail(policy, env, seed, deterministic):
    obs, _ = env.reset(seed=seed)
    ret, steps, crashed_at = 0.0, 0, None
    done = False
    while not done:
        x = torch.as_tensor(obs, dtype=torch.float32).flatten()
        if policy is None:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                p = policy(x)
            a = (int(torch.argmax(p).item()) if deterministic
                 else int(torch.multinomial(p, 1).item()))
        obs, r, te, tr, info = env.step(a)
        ret += float(r)
        steps += 1
        if info.get("crashed") and crashed_at is None:
            crashed_at = steps
        done = te or tr
    return ret, steps, crashed_at


def main():
    torch.manual_seed(0)
    policy = PolicyNetwork(25, 5)
    # train quickly using the same loop as record_demo
    sys.path.insert(0, str(ROOT / "experiments"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rd", ROOT / "experiments" / "record_demo.py")
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)

    train_env = gym.make("highway-v0", config=ENV_CONFIG)
    print("training 120 episodes (short, just to get a learning policy)...")
    rd.train(policy, train_env, 120, 0)
    train_env.close()

    env = gym.make("highway-v0", config=ENV_CONFIG)
    print()
    print("%-12s %10s %7s %10s" % ("policy", "return", "steps", "crashed_at"))
    print("-" * 42)
    for label, pol, det in (("untrained", None, False),
                            ("trained", policy, True)):
        rows = [rollout_detail(pol, env, 1000 + i, det) for i in range(10)]
        rets = [r[0] for r in rows]
        steps = [r[1] for r in rows]
        cats = [r[2] for r in rows]
        early = sum(1 for c, s in zip(cats, steps) if c is not None and c < 20)
        print("%-12s %10.2f %7.1f   %s" % (
            label, np.mean(rets), np.mean(steps),
            ",".join("–" if c is None else str(c) for c in cats)))
        print("%-12s   early crashes (<20 steps): %d/10" % ("", early))
    env.close()


if __name__ == "__main__":
    main()
