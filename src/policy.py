"""Policy network for REINFORCE on highway-env.

Architecture (Week 2 deliverable):
    flattened Kinematics observation (25 features)
        -> Linear(25, 128) -> ReLU
        -> Linear(128, 128) -> ReLU
        -> Linear(128, 5)   -> Softmax
    action sampled via Categorical distribution over the 5 meta-actions.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """MLP policy for highway-v0 (5 discrete actions).

    Parameters
    ----------
    obs_dim : int
        Flattened observation dimension (25 for default Kinematics).
    hidden_dim : int
        Width of hidden layers.
    n_actions : int
        Number of discrete actions (5 for highway-v0).
    """

    def __init__(
        self,
        obs_dim: int = 25,
        hidden_dim: int = 128,
        n_actions: int = 5,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return action probabilities P(a | s).

        Parameters
        ----------
        obs : Tensor of shape (obs_dim,) or (batch, obs_dim)

        Returns
        -------
        Tensor of shape (n_actions,) or (batch, n_actions) — a valid
        probability distribution (sums to 1 over the last dim).
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        logits = self.net(obs)
        return self.softmax(logits)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[int, torch.Tensor]:
        """Sample an action from the policy and return (action, log_prob).

        The log_prob retains its gradient graph so it can be used directly
        in the REINFORCE loss.  Use `deterministic=True` for inference /
        rollouts where you do not need gradients.

        Parameters
        ----------
        obs : Tensor (obs_dim,)
        deterministic : bool
            If True, pick the greedy argmax action instead of sampling.

        Returns
        -------
        action : int             — discrete action in {0..n_actions-1}
        log_prob : Tensor (1,)   — log π_θ(a | s), for the REINFORCE loss
        """
        probs = self(obs)  # (1, n_actions)
        if deterministic:
            return int(torch.argmax(probs, dim=-1).item()), torch.log(probs + 1e-8)[0]
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob

    def act_batch(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorised sampling for a batch of observations.

        Returns
        -------
        actions : LongTensor (batch,)
        log_probs : Tensor     (batch,)
        """
        probs = self(obs)
        dist = Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs
