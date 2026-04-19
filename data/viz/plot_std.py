"""Regenerate embedding-std plots for moco-v2 and barlow, matching the W&B loss curve style.

uv run python -m data.viz.plot_std
"""

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data.load_backbone import load_feature_extractor

_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_STATIC = Path(__file__).parent.parent.parent.parent / "personal-website/portfolio/static/clear-cxr"


class _ImageDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        return _TRANSFORM(Image.open(self.paths[idx]).convert("RGB"))


@torch.no_grad()
def _compute_std(outputs_dir: Path, data_dir: Path, n: int = 256) -> tuple[list[int], list[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted(data_dir.glob("images*/**/*.png"))
    paths = random.Random(42).sample(paths, min(n, len(paths)))
    loader = DataLoader(_ImageDataset(paths), batch_size=64, num_workers=4, pin_memory=True)

    by_epoch: dict[int, Path] = {}
    for p in outputs_dir.glob("epoch_*.pt"):
        parts = p.stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            by_epoch[int(parts[1])] = p

    epochs, stds = [], []
    for ep, ckpt_path in sorted(by_epoch.items()):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = load_feature_extractor(ckpt, device)
        feats = np.concatenate([model(imgs.to(device)).cpu().float().numpy() for imgs in loader])
        stds.append(float(feats.std(axis=0).mean()))
        epochs.append(ep)
        print(f"  epoch {ep:>4d}  std={stds[-1]:.4f}")

    return epochs, stds


def _plot(epochs: list[int], stds: list[float], title: str, out: Path) -> None:
    rc = {
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "#d0cec8",
        "axes.labelcolor":   "#555555",
        "axes.titlecolor":   "#333333",
        "axes.titlesize":    12,
        "axes.labelsize":    10,
        "axes.grid":         True,
        "grid.color":        "#e8e6e0",
        "grid.linewidth":    0.7,
        "grid.linestyle":    "--",
        "xtick.color":       "#888888",
        "ytick.color":       "#888888",
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "text.color":        "#555555",
        "savefig.facecolor": "white",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "lines.linewidth":   1.8,
        "lines.markersize":  5,
        "font.family":       "sans-serif",
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(6.58, 3.46))  # 632x332 at 96 dpi
        ax.plot(epochs, stds, "o-", color="#6b8cba")
        ax.set_ylabel("Mean feature std  (near 0 = collapsed)")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    data_dir = Path("datasets/nih-chest-xrays")

    print("MoCo v3 ...")
    epochs, stds = _compute_std(Path("outputs_cloud/moco-v2"), data_dir)
    _STATIC.mkdir(parents=True, exist_ok=True)
    _plot(epochs, stds, "MoCo v2: embedding std across training", _STATIC / "moco_std.png")

    print("\nBarlowTwins ...")
    epochs, stds = _compute_std(Path("outputs_cloud/barlow"), data_dir)
    _plot(epochs, stds, "BarlowTwins: embedding std across training", _STATIC / "barlow_std.png")

    print("\nSparK ...")
    epochs, stds = _compute_std(Path("outputs_cloud/spark"), data_dir)
    _plot(epochs, stds, "SparK: embedding std across training", _STATIC / "spark_std.png")

    print("\nDINO ...")
    epochs, stds = _compute_std(Path("outputs/dino"), data_dir)
    _plot(epochs, stds, "DINO: embedding std across training", _STATIC / "dino_std.png")


if __name__ == "__main__":
    main()
