# Notation Cheat Sheet

One-page reference for all symbols used across code, notes, and slides.
Keep this consistent everywhere — see `docs/highway_spec.md` for the
Kinematics observation matrix and action definitions, and
`docs/code-to-math.md` (Week 4, M2) for the mapping between these
symbols and actual code variables.

## MDP Fundamentals

| Symbol | Meaning | Code counterpart |
|---|---|---|
| $s \in \mathcal{S}$ | State (observation) | `obs` — Box `(5, 5)`, flatten to 25-dim for the network |
| $a \in \mathcal{A}$ | Action | `action` — `{0..4}` for `highway-v0` |
| $P(s' \mid s, a)$ | Transition probability of next state | handled by Gymnasium (environment) |
| $R(s, a)$ | Reward function | `reward` |
| $\gamma \in [0, 1)$ | Discount factor | `gamma` |
| $\rho(s_0)$ | Initial state distribution | Gymnasium reset |
| $r_t$ | Reward at time step $t$ | `rewards[t]` |

## Trajectories and Returns

| Symbol | Meaning | Code counterpart |
|---|---|---|
| $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_T)$ | Trajectory (episode) | one episode's recorded lists |
| $T$ | Episode length (number of steps) | `len(episode)` |
| $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k$ | Discounted return from time $t$ | `returns[t]` |
| $R(\tau) = \sum_{t=0}^{T-1} \gamma^{t} r_t = G_0$ | Total discounted return of a trajectory | `episode_return` |

> **Index convention (fixed deck-wide).** Reward $r_t$ is the reward received
> after taking action $a_t$ in state $s_t$. An episode of $T$ steps therefore
> has rewards $r_0, \dots, r_{T-1}$ and states $s_0, \dots, s_T$. Every sum in
> the deck runs over $t = 0, \dots, T-1$.

## Policy and Objective

| Symbol | Meaning | Code counterpart |
|---|---|---|
| $\theta$ | Policy parameters | `model.parameters()` |
| $\pi_\theta(a \mid s)$ | Policy: action probability given state | `probs` from softmax head |
| $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ | Score function | `log_prob` backward |
| $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$ | Expected return objective to maximize | mean of returns |
| $\mathcal{L}(\theta)$ | Loss to minimize (negative of above) | `loss` in training loop |

## Key Equations (LaTeX snippets)

```latex
\mathcal{S}, \mathcal{A}, P(s' \mid s, a), R(s, a), \gamma
\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_T), \quad G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)], \quad R(\tau) = \sum_{t=0}^{T-1} \gamma^{t} r_t = G_0
\nabla_\theta J(\theta) = \mathbb{E}_\tau \left[ \sum_{t=0}^{T-1} \gamma^{t} G_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]
```
