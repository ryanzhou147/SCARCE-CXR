"""Collapse monitor: tracks embedding diversity across training checkpoints.

Three metrics computed on a random sample of images at each saved checkpoint:

  - Mean feature std: average per-dimension std across the embedding matrix.
    Near 0 means all images map to nearly identical vectors (collapsed).

  - Mean pairwise cosine similarity: average similarity between random pairs.
    Near 1.0 means the encoder is outputting the same direction for everything.

  - Effective rank: exp(H) where H is the Shannon entropy of the normalised
    singular value spectrum of the feature matrix. Near 1 means the encoder
    is using only one linear direction (fully collapsed); near dim means it is
    using all dimensions equally.

Usage:
  uv run python -m data.collapse_monitor
  uv run python -m data.collapse_monitor --outputs-dir outputs/moco --n 512
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
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


def load_image_paths(data_dir: Path, n: int, seed: int) -> list[Path]:
    paths = sorted(data_dir.glob("images*/**/*.png"))
    return random.Random(seed).sample(paths, min(n, len(paths)))


@torch.no_grad()
def extract_features(model: nn.Module, paths: list[Path], device: torch.device) -> np.ndarray:
    dataset = ImageDataset(paths)
    loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
    feats = []
    for imgs in loader:
        feats.append(model(imgs.to(device)).cpu().float().numpy())
    return np.concatenate(feats)  # (N, dim) — raw backbone output, not normalised


def compute_metrics(feats: np.ndarray) -> tuple[float, float, float]:
    """Return (mean_feature_std, mean_cosine_sim, effective_rank)."""
    # 1. Mean per-dimension standard deviation
    mean_std = float(feats.std(axis=0).mean())

    # 2. Mean pairwise cosine similarity — sample random pairs for speed
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    normed = feats / (norms + 1e-8)
    n = len(normed)
    rng = np.random.default_rng(0)
    n_pairs = min(2048, n * (n - 1) // 2)
    idx_a = rng.integers(0, n, size=n_pairs)
    idx_b = rng.integers(0, n, size=n_pairs)
    valid = idx_a != idx_b
    cos_sims = (normed[idx_a[valid]] * normed[idx_b[valid]]).sum(axis=1)
    mean_cos = float(cos_sims.mean())

    # 3. Effective rank: exp(H(p)) where p is normalised singular value spectrum
    centered = feats - feats.mean(axis=0)
    _, sv, _ = np.linalg.svd(centered, full_matrices=False)
    sv = sv / sv.sum()
    sv = sv[sv > 0]
    eff_rank = float(np.exp(-np.sum(sv * np.log(sv))))

    return mean_std, mean_cos, eff_rank




def find_checkpoints(outputs_dir: Path) -> list[tuple[int, Path]]:
    """Return (epoch, path) pairs sorted by epoch, one entry per epoch."""
    by_epoch: dict[int, Path] = {}

    for p in outputs_dir.glob("epoch_*.pt"):
        try:
            ep = int(p.stem.split("_")[1])
            by_epoch[ep] = p
        except (IndexError, ValueError):
            continue

    # latest/best: read epoch from checkpoint; only add if not already covered
    for name in ("latest", "best"):
        p = outputs_dir / f"{name}.pt"
        if not p.exists():
            continue
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            ep = ckpt["epoch"] + 1
            if ep not in by_epoch:
                by_epoch[ep] = p
        except Exception:
            pass

    return sorted(by_epoch.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="Collapse monitor for CMVR checkpoints")
    parser.add_argument("--outputs-dir", type=str, default="outputs/barlow")
    parser.add_argument("--data-dir", type=str, default="datasets/nih-chest-xrays")
    parser.add_argument("--n", type=int, default=512, help="Images sampled per checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs_dir = Path(args.outputs_dir)
    data_dir = Path(args.data_dir)

    ckpts = find_checkpoints(outputs_dir)
    if not ckpts:
        print(f"No checkpoints found in {outputs_dir}")
        return
    print(f"Found {len(ckpts)} checkpoint(s): epochs {[e for e, _ in ckpts]}")

    paths = load_image_paths(data_dir, args.n, args.seed)
    print(f"Sampled {len(paths)} images (fixed across all checkpoints)\n")

    epochs, stds, cos_sims, eff_ranks = [], [], [], []
    for epoch, ckpt_path in ckpts:
        print(f"  Epoch {epoch:>4d}  ({ckpt_path.name})", end="  ", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        backbone = load_feature_extractor(ckpt, device)
        feats = extract_features(backbone, paths, device)
        std, cos, eff_rank = compute_metrics(feats)
        epochs.append(epoch)
        stds.append(std)
        cos_sims.append(cos)
        eff_ranks.append(eff_rank)
        print(f"std={std:.3f}  mean_cos={cos:.3f}  eff_rank={eff_rank:.1f}")

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(epochs, stds, "o-", color="#2196F3")
    axes[0].set_ylabel("Mean feature std")
    axes[0].set_title("Embedding diversity across training — collapse = all metrics flatten / drop")
    axes[0].grid(True, alpha=0.3)
    axes[0].annotate("↑ higher = more diverse", xy=(0.02, 0.88), xycoords="axes fraction",
                     fontsize=9, color="gray")

    axes[1].plot(epochs, cos_sims, "o-", color="#FF5722")
    axes[1].set_ylabel("Mean pairwise cosine sim")
    axes[1].grid(True, alpha=0.3)
    axes[1].annotate("↓ lower = more diverse", xy=(0.02, 0.88), xycoords="axes fraction",
                     fontsize=9, color="gray")

    axes[2].plot(epochs, eff_ranks, "o-", color="#4CAF50")
    axes[2].set_ylabel("Effective rank")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)
    axes[2].annotate("↑ higher = more diverse", xy=(0.02, 0.88), xycoords="axes fraction",
                     fontsize=9, color="gray")

    plt.tight_layout()
    run_method = method_name(torch.load(outputs_dir / "latest.pt", map_location="cpu", weights_only=False)) if (outputs_dir / "latest.pt").exists() else outputs_dir.name
    out = Path(f"{run_method}_collapse.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out}")
    plt.close()


if __name__ == "__main__":
    main()
