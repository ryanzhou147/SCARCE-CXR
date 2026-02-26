"""SparK — Sparse masked modeling for ResNet backbones.

Reference: Tian et al. 2023 "Designing BERT for Convolutional Networks:
Sparse and Hierarchical Masked Modeling" (ICLR 2023).

Why SparK over MAE for this project:
  - MAE's ViT needs large datasets (ImageNet scale) to learn good features.
  - SparK works with ResNet50, matching the backbone used by MoCo/DINO/BarlowTwins
    so all four methods produce directly comparable 2048-d features.
  - Convolutional inductive biases (translation equivariance, local connectivity)
    are well-matched to chest X-rays, which have consistent anatomy across images.

Architecture:
  Encoder  — ResNet50, split into 4 stages to extract hierarchical features.
             Masked patches are zeroed out before the first convolution, which
             degrades (but does not eliminate) information leakage compared to
             SparK's original sparse convolutions.  In practice, the masking
             signal is still strong enough for effective pretraining.
  Decoder  — Lightweight U-Net decoder with skip connections from all 4 encoder
             stages. Upsamples from 7×7 → 224×224 and predicts raw pixel values.
  Loss     — MSE on masked patches only; optional per-patch pixel normalisation
             (recommended — prevents trivial DC-component solutions).

At fine-tuning time the decoder is discarded; backbone features are extracted
via global average pool → 2048-d vector, identical to MoCo/DINO/BarlowTwins.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from torchvision import models


# ---------------------------------------------------------------------------
# Decoder building blocks
# ---------------------------------------------------------------------------

class _ConvUp(nn.Module):
    """ConvTranspose2d 2× upsample + BN + GELU."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _SparKDecoder(nn.Module):
    """Hierarchical U-Net decoder with skip connections from ResNet50 stages.

    Input feature map dimensions (ResNet50, image_size=224):
        f1: (B,  256, 56, 56)  — after layer1  (stride 4)
        f2: (B,  512, 28, 28)  — after layer2  (stride 8)
        f3: (B, 1024, 14, 14)  — after layer3  (stride 16)
        f4: (B, 2048,  7,  7)  — after layer4  (stride 32)

    Output: (B, 3, 224, 224) — full-resolution pixel predictions.
    """

    def __init__(self, dec_dim: int = 256):
        super().__init__()
        D = dec_dim

        # Project bottleneck to uniform decoder width
        self.proj4 = nn.Conv2d(2048, D, 1, bias=False)

        # Stage 4→3: 7×7 → 14×14, fuse with layer3 skip (1024 ch)
        self.up4   = _ConvUp(D, D)
        self.skip3 = nn.Conv2d(1024, D, 1, bias=False)
        self.fuse3 = nn.Sequential(nn.Conv2d(D * 2, D, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(D), nn.GELU())

        # Stage 3→2: 14×14 → 28×28, fuse with layer2 skip (512 ch)
        self.up3   = _ConvUp(D, D)
        self.skip2 = nn.Conv2d(512, D, 1, bias=False)
        self.fuse2 = nn.Sequential(nn.Conv2d(D * 2, D, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(D), nn.GELU())

        # Stage 2→1: 28×28 → 56×56, fuse with layer1 skip (256 ch)
        self.up2   = _ConvUp(D, D)
        self.skip1 = nn.Conv2d(256, D, 1, bias=False)
        self.fuse1 = nn.Sequential(nn.Conv2d(D * 2, D, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(D), nn.GELU())

        # Final upsample: 56×56 → 112×112 → 224×224
        self.up1 = _ConvUp(D, D // 2)
        self.up0 = _ConvUp(D // 2, D // 4)
        self.pred = nn.Conv2d(D // 4, 3, 1)

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
        f4: torch.Tensor,
    ) -> torch.Tensor:
        x = self.proj4(f4)                                     # (B, D, 7, 7)

        x = self.up4(x)                                        # (B, D, 14, 14)
        x = self.fuse3(torch.cat([x, self.skip3(f3)], dim=1)) # (B, D, 14, 14)

        x = self.up3(x)                                        # (B, D, 28, 28)
        x = self.fuse2(torch.cat([x, self.skip2(f2)], dim=1)) # (B, D, 28, 28)

        x = self.up2(x)                                        # (B, D, 56, 56)
        x = self.fuse1(torch.cat([x, self.skip1(f1)], dim=1)) # (B, D, 56, 56)

        x = self.up1(x)                                        # (B, D/2, 112, 112)
        x = self.up0(x)                                        # (B, D/4, 224, 224)
        return self.pred(x)                                    # (B, 3, 224, 224)


# ---------------------------------------------------------------------------
# SparK
# ---------------------------------------------------------------------------

class SparK(nn.Module):
    """SparK: masked autoencoder on a ResNet50 backbone.

    Args:
        img_size:       Input image resolution (square).
        patch_size:     Side length of each masked patch.  Must evenly divide
                        img_size.  32 → 7×7 grid (49 patches) for 224×224 input;
                        each 7×7 position in layer4 corresponds to one patch.
        encoder_name:   torchvision backbone identifier (``"resnet50"``).
        dec_dim:        Decoder hidden width.
        mask_ratio:     Fraction of patches to mask during pretraining.
        norm_pix_loss:  Normalise each patch before MSE (recommended).
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 32,
        encoder_name: str = "resnet50",
        dec_dim: int = 256,
        mask_ratio: float = 0.60,
        norm_pix_loss: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.n_patches = (img_size // patch_size) ** 2

        # ---- Encoder: split ResNet50 into stages ----------------------------
        backbone = getattr(models, encoder_name)(weights=None)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1   # stride 4  → (B,  256, 56, 56)
        self.layer2 = backbone.layer2   # stride 8  → (B,  512, 28, 28)
        self.layer3 = backbone.layer3   # stride 16 → (B, 1024, 14, 14)
        self.layer4 = backbone.layer4   # stride 32 → (B, 2048,  7,  7)
        self.avgpool = backbone.avgpool

        # ---- Decoder --------------------------------------------------------
        self.decoder = _SparKDecoder(dec_dim)

    # ---- Masking ------------------------------------------------------------

    def _random_mask(
        self, imgs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero out ``mask_ratio`` random patches.

        Returns:
            masked_imgs: same shape as imgs, zeroed at masked patches.
            mask_px:     (N, 1, H, W) float, 1 = masked, 0 = visible.
        """
        B, C, H, W = imgs.shape
        p = self.patch_size
        nh, nw = H // p, W // p
        n = nh * nw
        n_mask = int(n * self.mask_ratio)

        # Random patch shuffle per sample
        noise = torch.rand(B, n, device=imgs.device)
        ids = torch.argsort(noise, dim=1)
        mask = torch.zeros(B, n, device=imgs.device)
        mask.scatter_(1, ids[:, :n_mask], 1.0)  # 1 = masked

        # Upsample patch mask → pixel mask
        mask_px = mask.reshape(B, 1, nh, nw)
        mask_px = F.interpolate(mask_px, scale_factor=float(p), mode="nearest")

        masked_imgs = imgs * (1.0 - mask_px)
        return masked_imgs, mask_px

    # ---- Pixel normalisation ------------------------------------------------

    def _patchwise_normalize(self, imgs: torch.Tensor) -> torch.Tensor:
        """Normalise each spatial patch to zero-mean unit-variance.

        Prevents the loss from being dominated by absolute intensity
        (e.g. a bright vs. dark X-ray background).
        """
        B, C, H, W = imgs.shape
        p = self.patch_size
        nh, nw = H // p, W // p

        # (B, C, nh, p, nw, p) → (B, nh, nw, C, p, p)
        x = imgs.reshape(B, C, nh, p, nw, p).permute(0, 2, 4, 1, 3, 5)
        mean = x.mean(dim=(-1, -2, -3), keepdim=True)
        var  = x.var(dim=(-1, -2, -3), keepdim=True)
        x = (x - mean) / (var + 1e-6).sqrt()

        # Restore to (B, C, H, W)
        return x.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

    # ---- Forward / evaluation -----------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pretraining forward pass.

        Returns:
            loss:     Scalar MSE loss on masked patches.
            pred:     (N, 3, H, W) full-resolution pixel predictions.
            mask_px:  (N, 1, H, W) binary mask (1 = masked).
        """
        masked, mask_px = self._random_mask(x)

        # Encode — checkpoint each stage to avoid storing all skip-connection
        # feature maps simultaneously (f1+f2+f3+f4 would exceed L4 22GB)
        h = self.stem(masked)
        f1 = ckpt.checkpoint(self.layer1, h,  use_reentrant=False)
        f2 = ckpt.checkpoint(self.layer2, f1, use_reentrant=False)
        f3 = ckpt.checkpoint(self.layer3, f2, use_reentrant=False)
        f4 = ckpt.checkpoint(self.layer4, f3, use_reentrant=False)

        # Decode
        pred = self.decoder(f1, f2, f3, f4)  # (B, 3, H, W)

        target = self._patchwise_normalize(x) if self.norm_pix_loss else x
        loss = ((pred - target) ** 2 * mask_px).sum() / (mask_px.sum() * x.shape[1])
        return loss, pred, mask_px

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 2048-d backbone features for downstream use (no masking).

        Global-average-pooled layer4 output — identical interface to
        MoCo / DINO / BarlowTwins so the same eval scripts work unchanged.
        """
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        return self.avgpool(h).flatten(1)  # (N, 2048)
