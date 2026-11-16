"""State-value critic for REINFORCE with a learned baseline.

Task 39: "Implement state-value baseline critic network or running baseline
function."

What this is
------------
A second network V_phi(s) that predicts the expected return from state s
under the current policy. It is used as the baseline b(s) in

    grad J ≈ E[ (G_t - V_phi(s_t)) * grad log pi_theta(a_t | s_t) ]

The subtraction is free of bias because V_phi depends on s but **not** on the
action, so E_{a~pi}[grad log pi(a|s)] = 0 kills the extra term (see the
score-function property in the deck, §2). Correctness of the estimator does
not depend on the critic being accurate — only its variance does.

Why a separate network (and not the policy's own value head)
------------------------------------------------------------
The baseline lemma requires b to carry no theta-dependence, i.e. no gradient
path into the policy parameters. Giving the critic its own parameters phi and
detaching V_phi(s) when it is used as a weight keeps that requirement
mechanically true instead of relying on discipline.

Output layer
------------
No activation. A state value is an unbounded real number: it can be negative
when the reachable rewards under the policy are small compared to the current
position, and squashing it through ReLU/tanh would bias the regression.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """MLP critic V_phi(s) for highway-v0.

    Mirrors the policy trunk (two 128-wide ReLU layers) so the two networks
    have comparable capacity; only the head differs — one scalar instead of
    five logits.

    Parameters
    ----------
    obs_dim : int
        Flattened observation dimension (25 for the default Kinematics obs).
    hidden_dim : int
        Width of the hidden layers.
    """

    def __init__(self, obs_dim: int = 25, hidden_dim: int = 128) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict V(s).

        Parameters
        ----------
        obs : Tensor (obs_dim,) or (batch, obs_dim)

        Returns
        -------
        Tensor (1,) — or (batch, 1) for a batch. Kept as a trailing dim so
        callers can do `value.squeeze(-1)` and stay shape-explicit.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs)

    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        """V(s) with no gradient, as a flat (batch,) tensor.

        Used when the critic acts as a baseline weight: it must not carry a
        gradient path into phi, otherwise the policy update would try to
        differentiate through the critic as well.
        """
        with torch.no_grad():
            return self.forward(obs).squeeze(-1)


def critic_loss(values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean squared error between V_phi(s_t) and the sampled return G_t.

    Parameters
    ----------
    values : (T,) predictions for the states visited in one episode
    targets : (T,) reward-to-go G_t actually observed

    Returns
    -------
    Scalar loss. The targets are detached — they come from recorded rewards
    and carry no parameters, so differentiating them would be meaningless.
    """
    return nn.functional.mse_loss(values, targets.detach())
