"""Return estimators for REINFORCE.

Week 3 deliverables:
    - total_trajectory_return : G(τ) = sum_t γ^t r_t  (vanilla scalar return)
    - reward_to_go            : G_t = sum_{k=t}^{T} γ^{k-t} r_k  per-step

All functions accept plain Python lists or 1-D torch tensors of floats
and return 1-D torch tensors of matching length.
"""

from __future__ import annotations

from typing import Sequence, Union

import torch

Number = float
TensorLike = Union[Sequence[Number], torch.Tensor]


def _to_tensor(x: TensorLike, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Coerce a list or tensor into a 1-D float tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.tensor(list(x), dtype=dtype)
    """Coerce a list or tensor into a 1-D float tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.tensor(list(x), dtype=dtype)


def total_trajectory_return(
    rewards: TensorLike, gamma: float = 0.99
) -> torch.Tensor:
    """Discounted return of an entire trajectory.

    G(τ) = ∑_{t=0}^{T-1} γ^{t} r_t

    Parameters
    ----------
    rewards : sequence of float
        Per-step rewards r_0 ... r_{T-1}.
    gamma : float
        Discount factor in [0, 1].

    Returns
    -------
    Tensor of shape (1,) containing the scalar return for the episode.
    """
    r = _to_tensor(rewards)
    t = torch.arange(len(r), dtype=r.dtype)
    discounted = r * (gamma ** t)
    return discounted.sum().unsqueeze(0)


def reward_to_go(
    rewards: TensorLike, gamma: float = 0.99
) -> torch.Tensor:
    """Per-step reward-to-go (advantage baseline of zero).

    G_t = ∑_{k=t}^{T-1} γ^{k-t} r_k

    This is the standard REINFORCE reward-to-go estimator: each action at
    step t is weighted only by future rewards, which (for MDPs)
    reduces variance by ignoring past rewards that cannot be affected
    by that action.

    Parameters
    ----------
    rewards : sequence of float
        Per-step rewards r_0 ... r_{T-1}.
    gamma : float
        Discount factor in [0, 1].

    Returns
    -------
    Tensor of shape (T,) — G_t for each step t.
    """
    r = _to_tensor(rewards)
    T = len(r)
    returns = torch.zeros_like(r)
    # Walk backwards: G_t = r_t + γ·G_{t+1}
    g = torch.tensor(0.0, dtype=r.dtype)
    for t in reversed(range(T)):
        g = r[t] + gamma * g
        returns[t] = g
    return returns


def discounted_cumsum(
    x: torch.Tensor, gamma: float = 1.0
) -> torch.Tensor:
    """Generic backward discounted cumulative sum (helper).

    y_t = x_t + γ·x_{t+1} + γ²·x_{t+2} + ...
    """
    T = len(x)
    y = torch.zeros_like(x)
    acc = torch.tensor(0.0, dtype=x.dtype)
    for t in reversed(range(T)):
        acc = x[t] + gamma * acc
        y[t] = acc
    return y
