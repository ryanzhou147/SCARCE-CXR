"""Tracks embedding diversity across training checkpoints.

Three metrics on a random sample of images per checkpoint:
  - mean_std:    per-dimension std averaged across embedding dimensions. Near 0 = collapsed.
  - mean_cos:    average pairwise cosine similarity. Near 1.0 = collapsed.
  - eff_rank:    exp(H) where H is entropy of the singular value spectrum.
                 Near 1 = single direction used; near dim = all directions used equally.

uv run python -m data.eval.collapse_monitor --outputs-dir outputs/moco-v3
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data.load_backbone import load_feature_extractor, method_name

_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        return _TRANSFORM(Image.open(self.paths[idx]).convert("RGB"))


def load_image_paths(data_dir: Path, n: int) -> list[Path]:
    paths = sorted(data_dir.glob("images*/**/*.png"))
    return random.Random(42).sample(paths, min(n, len(paths)))


@torch.no_grad()
def extract_features(model: torch.nn.Module, paths: list[Path], device: torch.device) -> np.ndarray:
    loader = DataLoader(ImageDataset(paths), batch_size=64, num_workers=4, pin_memory=True)
    feats = [model(imgs.to(device)).cpu().float().numpy() for imgs in loader]
    return np.concatenate(feats)  # (N, dim) — unnormalised


def compute_metrics(feats: np.ndarray) -> tuple[float, float, float]:
    mean_std = float(feats.std(axis=0).mean())

    norms  = np.linalg.norm(feats, axis=1, keepdims=True)
    normed = feats / (norms + 1e-8)
    n      = len(normed)
    rng    = np.random.default_rng(0)
    n_pairs = min(2048, n * (n - 1) // 2)
    idx_a, idx_b = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
    valid    = idx_a != idx_b
    mean_cos = float((normed[idx_a[valid]] * normed[idx_b[valid]]).sum(axis=1).mean())

    centered = feats - feats.mean(axis=0)
    _, sv, _ = np.linalg.svd(centered, full_matrices=False)
    sv       = sv / sv.sum()
    sv       = sv[sv > 0]
    eff_rank = float(np.exp(-(sv * np.log(sv)).sum()))

    return mean_std, mean_cos, eff_rank


def find_checkpoints(outputs_dir: Path) -> list[tuple[int, Path]]:
    by_epoch: dict[int, Path] = {}
    for p in outputs_dir.glob("epoch_*.pt"):
        parts = p.stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            by_epoch[int(parts[1])] = p
    for name in ("latest", "best"):
        p = outputs_dir / f"{name}.pt"
        if p.exists():
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            ep = ckpt["epoch"] + 1
            if ep not in by_epoch:
                by_epoch[ep] = p
    return sorted(by_epoch.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=str, default="outputs/moco-v3")
    parser.add_argument("--data-dir", type=str, default="datasets/nih-chest-xrays")
    parser.add_argument("--n", type=int, default=512, help="Images sampled per checkpoint")
    args = parser.parse_args()

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs_dir = Path(args.outputs_dir)
    data_dir    = Path(args.data_dir)

    ckpts = find_checkpoints(outputs_dir)
    if not ckpts:
        print(f"No checkpoints found in {outputs_dir}")
        return
    print(f"Found {len(ckpts)} checkpoints: epochs {[e for e, _ in ckpts]}")

    paths = load_image_paths(data_dir, args.n)
    print(f"Using {len(paths)} fixed images across all checkpoints\n")

    epochs, stds, cos_sims, eff_ranks = [], [], [], []
    for epoch, ckpt_path in ckpts:
        print(f"  Epoch {epoch:>4d}  ({ckpt_path.name})", end="  ", flush=True)
        ckpt     = torch.load(ckpt_path, map_location=device, weights_only=False)
        backbone = load_feature_extractor(ckpt, device)
        feats    = extract_features(backbone, paths, device)
        std, cos, eff_rank = compute_metrics(feats)
        epochs.append(epoch)
        stds.append(std)
        cos_sims.append(cos)
        eff_ranks.append(eff_rank)
        print(f"std={std:.3f}  mean_cos={cos:.3f}  eff_rank={eff_rank:.1f}")

    _PLOT_RC = {
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.edgecolor":     "#d0cec8",
        "axes.labelcolor":    "#555555",
        "axes.titlecolor":    "#333333",
        "axes.titlesize":     12,
        "axes.labelsize":     10,
        "axes.grid":          True,
        "grid.color":         "#e8e6e0",
        "grid.linewidth":     0.7,
        "grid.linestyle":     "--",
        "xtick.color":        "#888888",
        "ytick.color":        "#888888",
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "text.color":         "#555555",
        "savefig.facecolor":  "white",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "lines.linewidth":    1.8,
        "lines.markersize":   5,
        "font.family":        "sans-serif",
    }
    plt.rcParams.update(_PLOT_RC)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(epochs, stds,      "o-", color="#6b8cba")
    axes[1].plot(epochs, cos_sims,  "o-", color="#c47b7b")
    axes[2].plot(epochs, eff_ranks, "o-", color="#7faa8e")
    axes[0].set_ylabel("Mean feature std  (↑ diverse)")
    axes[1].set_ylabel("Mean pairwise cos  (↓ diverse)")
    axes[2].set_ylabel("Effective rank  (↑ diverse)")
    axes[2].set_xlabel("Epoch")
    latest = outputs_dir / "latest.pt"
    run = method_name(torch.load(latest, map_location="cpu", weights_only=False)) if latest.exists() else outputs_dir.name
    axes[0].set_title(f"Collapse monitor: {run}", pad=10)
    plt.tight_layout()

    out = Path(f"{run}_collapse.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
