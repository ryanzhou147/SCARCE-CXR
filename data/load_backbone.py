"""Shared backbone loader for CMVR eval scripts.

Auto-detects the SSL method from the checkpoint config and returns a frozen
nn.Module where ``forward(x)`` → ``(B, feature_dim)`` backbone features.

Supports all four methods: MoCo, DINO, BarlowTwins, SparK.
"""

import torch
import torch.nn as nn


class _GetFeaturesWrapper(nn.Module):
    """Exposes a model's get_features() as its forward() method."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.get_features(x)


def load_feature_extractor(
    ckpt: dict,
    device: torch.device,
    random_init: bool = False,
) -> nn.Module:
    """Build a frozen feature extractor from a CMVR checkpoint.

    Args:
        ckpt:         Loaded checkpoint dict (must contain ``"config"`` key).
        device:       Target device.
        random_init:  If True, use the checkpoint architecture but skip loading
                      weights — creates an identical random-init baseline.

    Returns:
        Eval-mode, frozen nn.Module: ``forward(x)`` → ``(B, D)`` features.
    """
    config = ckpt["config"]

    if "moco" in config:
        from ssl_methods.moco.model import MoCo
        cfg = config["moco"]
        model = MoCo(
            encoder_name=cfg["encoder"],
            dim=cfg["dim"],
            K=cfg["queue_size"],
            m=cfg["momentum"],
            T=cfg["temperature"],
        )
        if not random_init:
            model.load_state_dict(ckpt["model"])
        model.encoder_q.fc = nn.Identity()
        model.encoder_k = None
        extractor = model.encoder_q

    elif "dino" in config:
        from ssl_methods.dino.model import DINO
        cfg = config["dino"]
        model = DINO(encoder_name=cfg["encoder"], out_dim=cfg["out_dim"])
        if not random_init:
            model.load_state_dict(ckpt["model"])
        extractor = model.student[0]  # ResNet backbone, fc=Identity

    elif "barlow" in config:
        from ssl_methods.barlow.model import BarlowTwins
        cfg = config["barlow"]
        model = BarlowTwins(
            encoder_name=cfg["encoder"],
            proj_dim=cfg["proj_dim"],
            proj_hidden_dim=cfg["proj_hidden_dim"],
        )
        if not random_init:
            model.load_state_dict(ckpt["model"])
        extractor = _GetFeaturesWrapper(model)

    elif "spark" in config:
        from ssl_methods.spark.model import SparK
        cfg = config["spark"]
        model = SparK(
            img_size=config["data"]["image_size"],
            patch_size=cfg["patch_size"],
            encoder_name=cfg["encoder"],
            dec_dim=cfg["dec_dim"],
            mask_ratio=cfg["mask_ratio"],
            norm_pix_loss=cfg.get("norm_pix_loss", True),
        )
        if not random_init:
            model.load_state_dict(ckpt["model"])
        extractor = _GetFeaturesWrapper(model)

    else:
        raise ValueError(
            f"Unrecognised SSL method. Config top-level keys: {list(config.keys())}"
        )

    extractor = extractor.to(device)
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad_(False)
    return extractor


def method_name(ckpt: dict) -> str:
    """Return a short method identifier for use in output filenames."""
    config = ckpt["config"]
    for key in ("moco", "dino", "barlow", "spark"):
        if key in config:
            return key
    return "unknown"
