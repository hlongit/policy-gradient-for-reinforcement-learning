# Highway-env `highway-v0` Specification

Reference table for the **default `highway-v0`** environment
(`highway_env.envs.highway.HighwayEnv`), verified live Oct 2026.

## Observation (Kinematics)

| Feature index | Vehicle feature | Units | Notes |
|---|---|---|---|
| 0 | `present` | 1 / 0 | 1 if this vehicle slot is occupied, else 0 |
| 1 | `x` (longitudinal) | m | distance along the road from ego |
| 2 | `y` (lateral) | m | lateral offset from ego's lane center |
| 3 | `vx` (longitudinal velocity) | m/s | |
| 4 | `vy` (lateral velocity) | m/s | |

The observation matrix is **5 × 5**: ego vehicle (row 0) + up to 4
neighbors (rows 1–4). Flatten to **25 features** for the MLP input layer
— i.e. `obs.reshape(-1)`.

```
obs = [[present, x, y, vx, vy],   # ego (row 0)
       [0,      x, y, vx, vy],    # neighbor 1
       ...]
```

## Action Space — 5 Discrete Meta-Actions

| Index | Meta-action | Effect |
|---|---|---|
| 0 | `LANE_LEFT` | move left one lane |
| 1 | `IDLE` | keep current lane & speed |
| 2 | `LANE_RIGHT` | move right one lane |
| 3 | `FASTER` | accelerate |
| 4 | `SLOWER` | decelerate |

## Reward & Termination

| Property | Value |
|---|---|
| Reward signal | Speed-based: +1 per step at target speed (20 m/s); collision penalty |
| Episode max length | 40 steps (default `duration`) |
| Termination | collision, off-road, or timeout |

## Config to Reproduce

```python
import gymnasium as gym
env = gym.make("highway-v0",
               render_mode="rgb_array",
               config={"observation": {"type": "Kinematics", "vehicles_count": 5},
                       "duration": 40})
```
