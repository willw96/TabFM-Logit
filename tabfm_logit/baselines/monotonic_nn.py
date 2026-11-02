"""
Monotonic neural network baseline.

A separable architecture
    V_k(x) = -g(cost_k(x)) + h_k(other_features(x))
where g is monotone non-decreasing in its 1-D input (so -g is monotone
non-increasing in cost) and h is an unconstrained MLP over the
remaining features. Choice probabilities follow the standard logit form
P_k = exp(V_k) / sum_j exp(V_j).

Monotonicity in g is enforced architecturally: weights are passed
through |·| at every step (equivalent to non-negative parameterization)
and the activation is ReLU. The composition of non-negative-weight
linear maps and ReLU is monotone non-decreasing in the input.

For alternatives that have no cost feature in the dataset spec
(LPMC walk/cycle), cost_k is set to zero. Since g(0) is the same
across all such alternatives, the contribution gets absorbed into
h_k's bias term and does not break the construction.

This baseline is foundation-model-independent: the same trained model
serves every (dataset) and produces the same numbers regardless of
whether it appears under the Mitra or TabPFN column in the headline
table.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def cost_indices_for(dataset: str, feature_names: list[str]) -> list[int | None]:
    """Return per-alternative cost feature indices, in alternative order.
    None for alternatives that have no cost feature."""
    if dataset == "swissmetro":
        # alts in order: train (0), sm (1), car (2)
        return [feature_names.index("TRAIN_CO"),
                feature_names.index("SM_CO"),
                feature_names.index("CAR_CO")]
    if dataset == "lpmc":
        # alts in order: walk (0), cycle (1), pt (2), drive (3)
        return [None,
                None,
                feature_names.index("cost_transit"),
                feature_names.index("cost_driving")]
    if dataset == "iot_wearables":
        return [feature_names.index(f"price{i+1}") for i in range(3)]
    raise ValueError(f"Unknown dataset for cost indices: {dataset}")


def split_features(X, cost_indices: list[int | None]):
    """Split X into (cost_per_alt, other_features) tensors.

    cost_per_alt: shape (N, K). cost[:, k] = X[:, cost_indices[k]] when
    cost_indices[k] is set, else zero.
    other_features: shape (N, D - K_with_cost). All features not in
    cost_indices.
    """
    n_obs, n_features = X.shape
    K = len(cost_indices)
    cost = torch.zeros((n_obs, K), dtype=X.dtype, device=X.device)
    used = set()
    for k, idx in enumerate(cost_indices):
        if idx is not None:
            cost[:, k] = X[:, idx]
            used.add(idx)
    other_idx = [i for i in range(n_features) if i not in used]
    other = X[:, other_idx]
    return cost, other


class MonotonicMLP(nn.Module):
    """Separable monotonic-in-cost choice model."""

    def __init__(self, n_alts: int, n_other_features: int,
                 hidden_g: int = 16, hidden_h: int = 32):
        super().__init__()
        self.n_alts = n_alts
        # g: monotone non-decreasing scalar function R -> R, shared across alts.
        # Weights are stored unconstrained; |·| is applied at forward time.
        self.g_W1 = nn.Parameter(torch.randn(1, hidden_g) * 0.1)
        self.g_b1 = nn.Parameter(torch.zeros(hidden_g))
        self.g_W2 = nn.Parameter(torch.randn(hidden_g, 1) * 0.1)
        self.g_b2 = nn.Parameter(torch.zeros(1))
        # h: unconstrained MLP over non-cost features, output K-dimensional.
        self.h = nn.Sequential(
            nn.Linear(n_other_features, hidden_h),
            nn.ReLU(),
            nn.Linear(hidden_h, n_alts),
        )

    def _g(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1). Returns (B, 1), monotone non-decreasing in x because
        # |W1|, |W2| are non-negative and ReLU is non-decreasing.
        h = F.relu(x @ torch.abs(self.g_W1) + self.g_b1)
        out = h @ torch.abs(self.g_W2) + self.g_b2
        return out

    def forward(self, cost: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        """cost: (N, K), other: (N, D'). Returns log-probabilities (N, K)."""
        N, K = cost.shape
        g_costs = self._g(cost.reshape(-1, 1)).reshape(N, K)
        V = -g_costs + self.h(other)
        return F.log_softmax(V, dim=1)
