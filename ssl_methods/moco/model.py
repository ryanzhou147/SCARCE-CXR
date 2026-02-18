"""MoCo v2 model: momentum encoder + ring-buffer queue of negative keys."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def build_encoder(encoder_name: str, dim: int) -> nn.Module:
    """ResNet backbone with a 2-layer MLP projection head (MoCo v2 style)."""
    backbone = getattr(models, encoder_name)(weights=None)
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Linear(feature_dim, feature_dim),
        nn.ReLU(),
        nn.Linear(feature_dim, dim),
    )
    return backbone


class MoCo(nn.Module):
    """MoCo v2: query encoder + momentum key encoder + negative-key queue.

    Args:
        encoder_name: torchvision ResNet name (e.g. ``"resnet50"``).
        dim:          Projection head output dimension.
        K:            Queue size — number of stored negative keys.
        m:            EMA coefficient for the momentum key encoder.
        T:            InfoNCE temperature.
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        dim: int = 128,
        K: int = 65536,
        m: float = 0.999,
        T: float = 0.07,
    ):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = build_encoder(encoder_name, dim)
        self.encoder_k = build_encoder(encoder_name, dim)

        # Key encoder starts as a copy of the query encoder; never gets gradients.
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False

        # Queue: shape (dim, K) — each column is one stored key vector.
        self.register_buffer("queue", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    # Internal helpers
    @torch.no_grad()
    def _momentum_update(self) -> None:
        """EMA update: key encoder slowly follows the query encoder."""
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """Overwrite the oldest entries in the queue ring buffer."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        end = ptr + batch_size
        if end <= self.K:
            self.queue[:, ptr:end] = keys.T
        else:
            # Wrap around.
            first = self.K - ptr
            self.queue[:, ptr:] = keys[:first].T
            self.queue[:, : end - self.K] = keys[first:].T
        self.queue_ptr[0] = end % self.K

    # Forward
    def forward(
        self, x_q: torch.Tensor, x_k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute InfoNCE logits and their ground-truth labels.

        Returns:
            logits: ``(N, 1+K)`` — column 0 is the positive pair.
            labels: ``(N,)`` zeros, so cross-entropy targets the positive.
        """
        q = F.normalize(self.encoder_q(x_q), dim=1)  # (N, dim)

        with torch.no_grad():
            self._momentum_update()
            k = F.normalize(self.encoder_k(x_k), dim=1)  # (N, dim)

        # Positive: each query dot its own key → (N, 1)
        l_pos = torch.einsum("nc,nc->n", q, k).unsqueeze(-1)
        # Negatives: each query dot every queued key → (N, K)
        l_neg = torch.einsum("nc,ck->nk", q, self.queue.clone().detach())

        logits = torch.cat([l_pos, l_neg], dim=1) / self.T  # (N, 1+K)
        labels = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)

        self._dequeue_and_enqueue(k)
        return logits, labels
