"""Display augmented views produced by MoCo or DINO data transforms.

Usage:
  uv run python -m data.show_augmented --method moco
  uv run python -m data.show_augmented --method dino
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

from data.dataset import collect_image_paths
from ssl_methods.moco.data import get_moco_transforms
from ssl_methods.dino.data import MultiCropTransform, get_dino_transforms

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
N_SAMPLES = 8


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    img = (tensor * _STD + _MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def sample_paths(config: dict, rng: random.Random) -> list[Path]:
    all_paths: list[Path] = []
    for ds_cfg in config["data"]["datasets"]:
        all_paths.extend(collect_image_paths(ds_cfg["name"], ds_cfg["root_dir"], ds_cfg.get("txt_file")))
    return rng.sample(all_paths, min(N_SAMPLES, len(all_paths)))


def show_moco(config: dict, paths: list[Path], out_path: Path) -> None:
    transform = get_moco_transforms(config)
    col_labels = ["Original", "View 1 (Query)", "View 2 (Key)"]
    fig, axes = plt.subplots(len(paths), 3, figsize=(7.5, len(paths) * 2.5))
    fig.suptitle("MoCo v2 Augmented Views", fontsize=14, y=1.01)

    for r, path in enumerate(paths):
        img = Image.open(path).convert("RGB")
        axrow = axes[r]
        axrow[0].imshow(img.convert("L"), cmap="gray")
        axrow[1].imshow(denormalize(transform(img)))
        axrow[2].imshow(denormalize(transform(img)))
        for c, ax in enumerate(axrow):
            ax.axis("off")
            if r == 0:
                ax.set_title(col_labels[c], fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    print(f"Saved grid to {out_path}")
    plt.show()


def show_dino(config: dict, paths: list[Path], out_path: Path) -> None:
    global_transform, local_transform = get_dino_transforms(config)
    dino_cfg = config["dino"]
    n_global = dino_cfg.get("n_global_crops", 2)
    n_local = dino_cfg.get("n_local_crops", 6)
    multicrop = MultiCropTransform(global_transform, local_transform, n_global, n_local)

    col_labels = (
        ["Original"]
        + [f"Global {i + 1}" for i in range(n_global)]
        + [f"Local {i + 1}" for i in range(n_local)]
    )
    n_cols = len(col_labels)
    fig, axes = plt.subplots(len(paths), n_cols, figsize=(n_cols * 2, len(paths) * 2))
    fig.suptitle("DINO Multi-Crop Augmented Views", fontsize=14, y=1.01)

    for r, path in enumerate(paths):
        img = Image.open(path).convert("RGB")
        views = multicrop(img)
        axrow = axes[r]
        axrow[0].imshow(img.convert("L"), cmap="gray")
        for c, view in enumerate(views, start=1):
            axrow[c].imshow(denormalize(view))
        for c, ax in enumerate(axrow):
            ax.axis("off")
            if r == 0:
                ax.set_title(col_labels[c], fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    print(f"Saved grid to {out_path}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise SSL data augmentations")
    parser.add_argument("--method", choices=["moco", "dino"], required=True)
    args = parser.parse_args()

    with open(f"configs/{args.method}.yaml") as f:
        config = yaml.safe_load(f)

    paths = sample_paths(config, random.Random(42))
    out_path = Path(f"augmented_{args.method}.png")

    if args.method == "moco":
        show_moco(config, paths, out_path)
    else:
        show_dino(config, paths, out_path)


if __name__ == "__main__":
    main()
