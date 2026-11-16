"""Re-check the 10-seed evaluation that reported crashes 10/10 for a trained
policy whose mean return was 17.

The two facts are only compatible if the crashes happen late (near the 40-step
limit). This trains the SAME configuration record_demo uses (300 episodes,
seed 0) and prints, per seed, the return, the step count and the step at which
a collision occurred. If every crash is on the final step, `crashed=True` is
technically correct and completely misleading as a summary.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import importlib.util
import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch

from src.policy import PolicyNetwork

ENV_CONFIG = {"vehicles_count": 20}
EPOCHS = 300


def detail(policy, env, seed, deterministic):
    obs, _ = env.reset(seed=seed)
    ret, steps, crash_step = 0.0, 0, None
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
        if info.get("crashed") and crash_step is None:
            crash_step = steps
        done = te or tr
    return ret, steps, crash_step


def main():
    spec = importlib.util.spec_from_file_location(
        "rd", ROOT / "experiments" / "record_demo.py")
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)

    torch.manual_seed(0)
    policy = PolicyNetwork(25, 5)
    tenv = gym.make("highway-v0", config=ENV_CONFIG)
    print(f"training {EPOCHS} episodes (exactly what record_demo does)...")
    hist = rd.train(policy, tenv, EPOCHS, 0)
    tenv.close()
    print(f"  last25 mean {np.mean(hist[-25:]):.2f}")
    print()

    env = gym.make("highway-v0", config=ENV_CONFIG)
    print("%-4s %8s %6s %10s %s" % ("seed", "return", "steps", "crash_step", "note"))
    print("-" * 50)
    n_crash = n_early = 0
    for i in range(10):
        s = 1000 + i
        ret, steps, cs = detail(policy, env, s, True)
        n_crash += cs is not None
        n_early += (cs is not None and cs < 35)
        note = ("crash on final step" if cs == steps
                else ("early crash" if cs is not None else "no crash"))
        print("%-4d %8.2f %6d %10s %s" % (s, ret, steps, cs if cs else "–", note))
    env.close()

    print()
    print(f"crashed at all      : {n_crash}/10")
    print(f"crashed before step 35: {n_early}/10")
    print()
    print("If almost all crashes land on the final step, the boolean summary")
    print("'crashes 10/10' is true but reads as 'the agent cannot drive'.")


if __name__ == "__main__":
    main()
