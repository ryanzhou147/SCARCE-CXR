"""
SparK reconstruction GradCAM: shows which visible patches the encoder attends to
when reconstructing masked regions. The backward signal is reconstruction MSE on
masked patches only, not a classifier score.

Columns: original | masked | layer2 CAM | layer3 CAM | layer4 CAM | composite

Usage:
  uv run python -m data.viz.spark_recon_gradcam --checkpoint outputs_cloud/spark/best.pt
  uv run python -m data.viz.spark_recon_gradcam --checkpoint outputs_cloud/spark/best.pt --n-images 4
"""

import argparse
import random
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms

from data.dataloader import collect_image_paths
from ssl_methods.spark.model import SparK

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

LAYERS = ["layer1", "layer2", "layer3", "layer4"]
LAYER_LABELS = {
    "layer1": "layer1\nedges/contrast",
    "layer2": "layer2\n(frozen in FT)",
    "layer3": "layer3\n(unfrozen)",
    "layer4": "layer4\n(unfrozen)",
}


def _denorm(t: torch.Tensor) -> np.ndarray:
    return (t * _STD + _MEAN).clamp(0, 1).mean(0).numpy()


class _ReconGradCAM:
    def __init__(self, model: SparK):
        self.activations: dict[str, torch.Tensor] = {}
        self.gradients: dict[str, torch.Tensor] = {}
        for name in LAYERS:
            layer = getattr(model, name)
            layer.register_forward_hook(self._act_hook(name))
            layer.register_full_backward_hook(self._grad_hook(name))

    def _act_hook(self, name: str):
        def hook(module, inp, out):
            self.activations[name] = out
        return hook

    def _grad_hook(self, name: str):
        def hook(module, grad_in, grad_out):
            self.gradients[name] = grad_out[0]
        return hook

    def compute(
        self,
        model: SparK,
        img_tensor: torch.Tensor,
        mask_seed: int = 42,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, int, int]:
        """Run forward+backward targeting a single masked patch.

        Returns (cams, original_np, masked_np, target_row, target_col).
        cams keys: "layer1", "layer2", "layer3", "layer4" — each (224, 224) in [0, 1].
        target_row/col are patch-grid coordinates of the explained patch.
        """
        p = model.patch_size
        nh = nw = img_tensor.shape[-1] // p

        torch.manual_seed(mask_seed)
        masked, mask_px = model._random_mask(img_tensor)

        # Pick one masked patch deterministically using the seed
        patch_grid = mask_px[0, 0, ::p, ::p].cpu().numpy()
        masked_coords = [(r, c) for r in range(nh) for c in range(nw) if patch_grid[r, c] > 0.5]
        rng = random.Random(mask_seed)
        target_row, target_col = rng.choice(masked_coords)

        # Build a pixel mask covering only that one patch
        single_px = torch.zeros_like(mask_px)
        single_px[:, :, target_row * p:(target_row + 1) * p, target_col * p:(target_col + 1) * p] = 1.0

        model.zero_grad()
        with torch.enable_grad():
            h  = model.stem(masked)
            f1 = model.layer1(h)
            f2 = model.layer2(f1)
            f3 = model.layer3(f2)
            f4 = model.layer4(f3)
            pred = model.decoder(f1, f2, f3, f4)

            target = model._patchwise_normalize(img_tensor) if model.norm_pix_loss else img_tensor
            loss = ((pred - target) ** 2 * single_px).sum() / (single_px.sum() * img_tensor.shape[1])
            loss.backward()

        cams: dict[str, np.ndarray] = {}
        for name in LAYERS:
            weights = self.gradients[name].mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * self.activations[name]).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
            cam = cam.squeeze().detach().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cams[name] = cam

        orig_np   = _denorm(img_tensor[0].cpu())
        masked_np = _denorm(masked[0].cpu())

        return cams, orig_np, masked_np, target_row, target_col


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="datasets/nih-chest-xrays")
    parser.add_argument("--n-images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading checkpoint...")
    ckpt_data = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sp_cfg = ckpt_data["config"]["spark"]
    cfg = ckpt_data["config"]
    epoch = ckpt_data.get("epoch", "?")

    model = SparK(
        img_size=cfg["data"]["image_size"],
        patch_size=sp_cfg["patch_size"],
        encoder_name=sp_cfg["encoder"],
        dec_dim=sp_cfg["dec_dim"],
        mask_ratio=sp_cfg["mask_ratio"],
        norm_pix_loss=sp_cfg.get("norm_pix_loss", True),
    )
    model.load_state_dict(ckpt_data["model"])
    model.train(False).to(device)
    print(f"  SparK ep{epoch}")

    gcam = _ReconGradCAM(model)

    all_paths = collect_image_paths("nih", args.data_dir)
    random.seed(args.seed)
    paths = random.sample(all_paths, args.n_images)

    n = len(paths)
    ncols = 1 + len(LAYERS)  # original, layer cams
    fig, axes = plt.subplots(n, ncols, figsize=(ncols * 3, n * 3))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["masked (60%)"] + [LAYER_LABELS[l] for l in LAYERS]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=10)

    for i, path in enumerate(paths):
        img_tensor = _TRANSFORM(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

        result = gcam.compute(model, img_tensor, mask_seed=args.seed + i)
        if result is None:
            continue
        cams, original_np, masked_np, trow, tcol = result

        p = model.patch_size
        box = mpatches.Rectangle(
            (tcol * p, trow * p), p, p,
            linewidth=2, edgecolor="#facc15", facecolor="none",
        )
        axes[i, 0].imshow(masked_np, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].add_patch(box)
        for j, name in enumerate(LAYERS):
            axes[i, 1 + j].imshow(original_np, cmap="gray", vmin=0, vmax=1)
            axes[i, 1 + j].imshow(cams[name], cmap="jet", alpha=0.45, vmin=0, vmax=1)
        for ax in axes[i]:
            ax.axis("off")

    fig.suptitle(
        "Layer-by-layer Grad-CAM: reconstruction signal (SparK)",
        fontsize=18, fontweight="bold", color="#333333",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    _STATIC = Path(__file__).parent.parent.parent.parent / "personal-website/portfolio/static/clear-cxr"
    out = Path(args.out) if args.out else _STATIC / "spark_recon_gradcam.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.close()


if __name__ == "__main__":
    main()
