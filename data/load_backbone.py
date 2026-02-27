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


class _SparKBackbone(nn.Module):
    """Exposes SparK encoder stages as a ResNet-like backbone.

    Copies the stage references so .layer1/.layer2/.layer3/.layer4 are named
    children — required for selective layer unfreezing in fine-tuning.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.stem    = model.stem
        self.layer1  = model.layer1
        self.layer2  = model.layer2
        self.layer3  = model.layer3
        self.layer4  = model.layer4
        self.avgpool = model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        return self.avgpool(h).flatten(1)


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


def load_raw_backbone(
    ckpt: dict,
    device: torch.device,
    random_init: bool = False,
) -> tuple[nn.Module, str]:
    """Return ``(backbone, method)`` for gradient-based fine-tuning.

    Unlike ``load_feature_extractor``, the returned backbone has named ResNet
    stages (``.layer1`` / ``.layer2`` / ``.layer3`` / ``.layer4``) as direct
    children so callers can selectively unfreeze them.

    All parameters are frozen by default.  Call ``unfreeze_for_finetuning``
    to open the appropriate layers based on ``method``.

    ``method`` is one of: ``"moco"``, ``"dino"``, ``"barlow"``, ``"spark"``.
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
        backbone = model.encoder_q
        backbone.fc = nn.Identity()
        method = "moco"

    elif "dino" in config:
        from ssl_methods.dino.model import DINO
        cfg = config["dino"]
        model = DINO(encoder_name=cfg["encoder"], out_dim=cfg["out_dim"])
        if not random_init:
            model.load_state_dict(ckpt["model"])
        backbone = model.student[0]
        method = "dino"

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
        backbone = model.backbone   # torchvision ResNet with fc=Identity
        method = "barlow"

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
        backbone = _SparKBackbone(model)
        method = "spark"

    else:
        raise ValueError(
            f"Unrecognised SSL method. Config top-level keys: {list(config.keys())}"
        )

    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone, method


def unfreeze_for_finetuning(backbone: nn.Module, method: str) -> list[str]:
    """Selectively unfreeze ResNet stages for gradient-based fine-tuning.

    Based on surgical fine-tuning literature:
      - Contrastive methods (MoCo, BarlowTwins, DINO): unfreeze layer2 onwards.
        These methods learn strong abstract features in early layers; mid-to-late
        layers benefit from domain adaptation.
      - Restorative methods (SparK): unfreeze layer3 onwards.
        The decoder forces layer3/4 to encode fine-grained spatial detail;
        layers 1/2 already contain domain-invariant texture features.

    BatchNorm layers are always kept frozen regardless of stage — small
    fine-tuning datasets would corrupt the pretraining statistics.

    Returns the list of unfrozen stage names.
    """
    if method in ("moco", "barlow", "dino"):
        unfreeze = {"layer2", "layer3", "layer4"}
    else:   # spark — restorative
        unfreeze = {"layer3", "layer4"}

    for name, child in backbone.named_children():
        for p in child.parameters():
            p.requires_grad_(name in unfreeze)

    # Always freeze BatchNorm — preserve pretraining statistics
    for m in backbone.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            for p in m.parameters():
                p.requires_grad_(False)

    return sorted(unfreeze)


def method_name(ckpt: dict) -> str:
    """Return a short method identifier for use in output filenames."""
    config = ckpt["config"]
    for key in ("moco", "dino", "barlow", "spark"):
        if key in config:
            return key
    return "unknown"
