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
| Reward signal | **normalised mixture**, $r \in [0,1]$ (see below) |
| `high_speed_reward` | $0.4$ |
| `right_lane_reward` | $0.1$ |
| `collision_reward` | $-1$ |
| `lane_change_reward` | $0$ |
| `reward_speed_range` | $[20, 30]$ m/s |
| `normalize_reward` | `True` |
| Episode max length | 40 steps (default `duration`) |
| Termination | collision, off-road, or timeout |

The reward is **not** a flat $+1$ at a target speed. `HighwayEnv._reward`
computes a weighted sum of the named reward terms, linearly maps it from
`[collision_reward, high_speed_reward + right_lane_reward]` $= [-1, 0.5]$ onto
$[0, 1]$, then multiplies by the on-road indicator:

```python
reward = utils.lmap(reward, [collision_reward,
                             high_speed_reward + right_lane_reward], [0, 1])
reward *= rewards["on_road_reward"]
```

Consequence for the slides: since $r_t \le 1$ and $\gamma = 0.99$,
$G_0 \le \sum_{k<40} 0.99^{k} = 33.10$, so gradient weights are bounded.

## Config to Reproduce

```python
import gymnasium as gym
env = gym.make("highway-v0",
               render_mode="rgb_array",
               config={"observation": {"type": "Kinematics", "vehicles_count": 5},
                       "duration": 40})
```
