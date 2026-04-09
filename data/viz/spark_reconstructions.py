"""
Visualise SparK masked image reconstructions  
uv run python -m data.viz.spark_reconstructions --checkpoint outputs/spark/latest.pt
uv run python -m data.viz.spark_reconstructions --checkpoint outputs/spark/latest.pt --n-images 8 --output recon.png
"""

import random
from pathlib import Path

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms

from data.dataloader import collect_image_paths
from ssl_methods.spark.model import SparK

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _denorm(t: torch.Tensor) -> np.ndarray:
    return (t * _STD + _MEAN).clamp(0, 1).mean(0).numpy()



if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Visualise SparK reconstructions")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="datasets/nih-chest-xrays")
    parser.add_argument("--n-images", type=int, default=6)
    parser.add_argument("--sample-index", type=int, default=None, help="Pick one specific image by index (overrides --n-images)")
    parser.add_argument("--output", default="spark_reconstructions.png")
    args = parser.parse_args()

    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    sp_cfg = config["spark"]

    model = SparK(
        img_size=config["data"]["image_size"],
        patch_size=sp_cfg["patch_size"],
        encoder_name=sp_cfg["encoder"],
        dec_dim=sp_cfg["dec_dim"],
        mask_ratio=sp_cfg["mask_ratio"],
        norm_pix_loss=False,
    )
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    print(f"  Checkpoint epoch: {ckpt.get('epoch', '?')}")

    paths = collect_image_paths("nih", args.data_dir)
    paths = random.sample(paths, min(100, len(paths)))  # sample pool
    if args.sample_index is not None:
        paths = [paths[args.sample_index]]
    else:
        paths = paths[:args.n_images]

    size = config["data"]["image_size"]
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    imgs = torch.stack([
        transform(Image.open(p).convert("RGB")) for p in paths
    ]).to(device)

    with torch.no_grad():
        _, pred, mask_px = model(imgs)

    imgs    = imgs.cpu()
    pred    = pred.cpu()
    mask_px = mask_px.cpu()

    n = len(paths)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = axes[None]

    for col, title in enumerate(["Original", "Composite"]):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for i in range(n):
        orig_full = _denorm(imgs[i])
        mask      = mask_px[i, 0].numpy()

        # Denormalise prediction using the global statistics of the visible pixels.
        # The model predicts in per-patch-normalised space (each patch ~N(0,1)).
        # Applying a single global mean/std maps all patches to a consistent
        # brightness reference frame, eliminating inter-patch seams caused by
        # independent per-patch statistics.
        vis_pixels = orig_full[mask == 0]
        global_mean = float(vis_pixels.mean()) if vis_pixels.size > 0 else float(orig_full.mean())
        global_std  = float(vis_pixels.std())  if vis_pixels.size > 0 else float(orig_full.std())
        global_std  = max(global_std, 1e-8)

        pred_gray     = pred[i].mean(0).numpy()
        recon_display = (pred_gray * global_std + global_mean).clip(0, 1)

        # Feather mask edges for composite
        recon_smooth = gaussian_filter(recon_display, sigma=2.0)
        soft_mask    = gaussian_filter(mask.astype(np.float32), sigma=3.0)
        composite    = orig_full * (1 - soft_mask) + recon_smooth * soft_mask

        for col, display_img in enumerate([orig_full, composite]):
            axes[i, col].imshow(display_img, cmap="gray", vmin=0, vmax=1, aspect="auto")
            axes[i, col].axis("off")

    plt.tight_layout()
    out = Path(args.output)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.show()

