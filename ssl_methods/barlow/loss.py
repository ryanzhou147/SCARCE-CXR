"""BarlowTwins loss: cross-correlation matrix between two views' projections.

Reference: Zbontar et al. 2021 "Barlow Twins: Self-Supervised Learning via
Redundancy Reduction" (https://arxiv.org/abs/2103.03230)

The loss has two terms:
  - Invariance:  diagonal of C should be 1  (same image → same embedding)
  - Decorrelation: off-diagonal of C should be 0  (different dims independent)

C_ij = (sum_b z1_bi · z2_bj) / N  after batch-normalising z1 and z2.

No teacher, no queue, no centering buffer — the cross-correlation objective
directly prevents both mode collapse and uniform collapse.
"""

import torch
import torch.nn as nn


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Return a flattened view of the off-diagonal elements of a square matrix."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinsLoss(nn.Module):
    """Cross-correlation loss for Barlow Twins.

    Args:
        lambda_coeff: Weight on the off-diagonal (redundancy-reduction) term.
                      Paper uses 0.005; lower values put more emphasis on
                      invariance, higher values on decorrelation.
    """

    def __init__(self, lambda_coeff: float = 0.005):
        super().__init__()
        self.lambda_coeff = lambda_coeff

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute BarlowTwins loss.

        Args:
            z1: (N, proj_dim) projections from view 1.
            z2: (N, proj_dim) projections from view 2.

        Returns:
            Scalar loss.
        """
        N = z1.size(0)

        # Batch-normalise along the sample dimension so C is a true correlation matrix.
        z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
        z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)

        # Cross-correlation matrix: (proj_dim, proj_dim)
        c = z1.T @ z2 / N

        # Invariance: pull diagonal toward 1
        on_diag = (torch.diagonal(c) - 1).pow(2).sum()
        # Redundancy reduction: push off-diagonal toward 0
        off_diag = _off_diagonal(c).pow(2).sum()

        return on_diag + self.lambda_coeff * off_diag
