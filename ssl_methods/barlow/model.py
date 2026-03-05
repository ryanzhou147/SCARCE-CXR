"""
BarlowTwins model: shared backbone + 3-layer BatchNorm projector.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt
from torchvision import models


def _build_backbone(encoder_name: str) -> tuple[nn.Module, int]:
    """Return (backbone, feature_dim) with the classifier head replaced by Identity."""
    if encoder_name.startswith("vit"):
        backbone = getattr(models, encoder_name)(weights=None)
        feature_dim: int = backbone.hidden_dim
        backbone.heads = nn.Identity()
    else:
        backbone = getattr(models, encoder_name)(weights=None)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    return backbone, feature_dim


class BarlowTwins(nn.Module):
    """BarlowTwins student model.

    Both views pass through the same backbone and projector.

    Args:
        encoder_name:     torchvision backbone (e.g. "resnet50").
        proj_dim:         Output dimension of the projector.
        proj_hidden_dim:  Hidden width of the projector MLP.
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        proj_dim: int = 2048,
        proj_hidden_dim: int = 2048,
    ):
        super().__init__()
        self.backbone, feature_dim = _build_backbone(encoder_name)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_dim, bias=False),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone with gradient checkpointing on each ResNet layer block."""
        bb = self.backbone
        x = bb.conv1(x)
        x = bb.bn1(x)
        x = bb.relu(x)
        x = bb.maxpool(x)
        x = ckpt.checkpoint(bb.layer1, x, use_reentrant=False)
        x = ckpt.checkpoint(bb.layer2, x, use_reentrant=False)
        x = ckpt.checkpoint(bb.layer3, x, use_reentrant=False)
        x = ckpt.checkpoint(bb.layer4, x, use_reentrant=False)
        x = bb.avgpool(x)
        return torch.flatten(x, 1)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (z1, z2) projections for both views, shape (N, proj_dim)."""
        z1 = self.projector(self._encode(x1))
        z2 = self.projector(self._encode(x2))
        return z1, z2

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return (N, feature_dim) backbone features; projector is discarded."""
        return self.backbone(x)
