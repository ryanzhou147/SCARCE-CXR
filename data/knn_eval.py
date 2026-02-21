"""k-NN evaluation for MoCo pretrained encoder.

For each validation image, finds the k nearest neighbours in the training
set by cosine similarity and predicts the majority-vote label. Requires no
training — an instant, assumption-free quality check of the embedding space.

Sweeps k = [1, 5, 10, 20, 50] and compares pretrained vs random-init baseline.

Usage:
  uv run python -m data.knn_eval
  uv run python -m data.knn_eval --checkpoint outputs/moco/best.pt --k 1 5 20
"""

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ssl_methods.moco.model import MoCo

DEFAULT_KS = [1, 5, 10, 20, 50]
MIN_TEST_PER_CLASS = 10

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


def load_split(data_dir: Path, txt_file: str) -> set[str]:
    return set((data_dir / txt_file).read_text().strip().splitlines())


def load_single_label_samples(
    data_dir: Path, split_names: set[str]
) -> dict[str, list[Path]]:
    index = {p.name: p for p in data_dir.glob("images*/**/*.png")}
    label_to_paths: dict[str, list[Path]] = {}
    with open(data_dir / "Data_Entry_2017.csv") as f:
        for row in csv.DictReader(f):
            name = row["Image Index"]
            if name not in split_names or name not in index:
                continue
            labels = row["Finding Labels"].split("|")
            if len(labels) != 1:
                continue
            label_to_paths.setdefault(labels[0], []).append(index[name])
    return label_to_paths


@torch.no_grad()
def extract_features(model: nn.Module, paths: list[Path], device: torch.device) -> np.ndarray:
    dataset = ImageDataset(paths)
    loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
    feats = []
    for imgs in loader:
        feats.append(model(imgs.to(device)).cpu().numpy())
    return normalize(np.concatenate(feats))


def knn_predict(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    val_feats: np.ndarray,
    k: int,
) -> np.ndarray:
    """Cosine k-NN with majority vote. Both feature matrices must be L2-normalised."""
    sim = val_feats @ train_feats.T          # (n_val, n_train)
    top_k = np.argsort(-sim, axis=1)[:, :k]  # (n_val, k)
    top_k_labels = train_labels[top_k]        # (n_val, k)
    return np.array([np.bincount(row).argmax() for row in top_k_labels])


def build_backbone(config: dict, checkpoint: dict | None, device: torch.device) -> nn.Module:
    moco_cfg = config["moco"]
    model = MoCo(
        encoder_name=moco_cfg["encoder"],
        dim=moco_cfg["dim"],
        K=moco_cfg["queue_size"],
        m=moco_cfg["momentum"],
        T=moco_cfg["temperature"],
    ).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
    model.encoder_q.fc = nn.Identity()
    model.encoder_k = None
    model.eval()
    return model.encoder_q


def main() -> None:
    parser = argparse.ArgumentParser(description="k-NN evaluation for MoCo")
    parser.add_argument("--checkpoint", type=str, default="outputs/moco/best.pt")
    parser.add_argument("--data-dir", type=str, default="datasets/nih-chest-xrays")
    parser.add_argument("--k", type=int, nargs="+", default=DEFAULT_KS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    epoch = checkpoint["epoch"] + 1
    print(f"Checkpoint epoch: {epoch}")

    train_names = load_split(data_dir, "train.txt")
    val_names = load_split(data_dir, "val.txt")

    print("Loading single-label images...")
    train_by_label = load_single_label_samples(data_dir, train_names)
    val_by_label = load_single_label_samples(data_dir, val_names)

    classes = sorted(
        c for c in train_by_label
        if c in val_by_label and len(val_by_label[c]) >= MIN_TEST_PER_CLASS
    )
    print(f"Classes ({len(classes)}): {classes}")
    label_to_idx = {c: i for i, c in enumerate(classes)}

    train_paths = [p for c in classes for p in train_by_label[c]]
    train_labels_raw = [c for c in classes for _ in train_by_label[c]]
    val_paths = [p for c in classes for p in val_by_label[c]]
    val_labels_raw = [c for c in classes for _ in val_by_label[c]]

    train_labels = np.array([label_to_idx[l] for l in train_labels_raw])
    val_labels = np.array([label_to_idx[l] for l in val_labels_raw])
    print(f"Train: {len(train_paths)} | Val: {len(val_paths)}")

    results: dict[str, dict[int, float]] = {}
    for name, ckpt in [("Pretrained", checkpoint), ("Random init", None)]:
        print(f"\nExtracting features — {name}...")
        backbone = build_backbone(config, ckpt, device)
        train_feats = extract_features(backbone, train_paths, device)
        val_feats = extract_features(backbone, val_paths, device)

        accs: dict[int, float] = {}
        for k in sorted(args.k):
            preds = knn_predict(train_feats, train_labels, val_feats, k)
            acc = float((preds == val_labels).mean())
            accs[k] = acc
            print(f"  {k:3d}-NN: {acc * 100:.1f}%")
        results[name] = accs

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Pretrained": "#2196F3", "Random init": "#FF5722"}
    for name, accs in results.items():
        ks = sorted(accs)
        ax.plot(ks, [accs[k] * 100 for k in ks], marker="o", label=name, color=colors[name])

    ax.set_xlabel("k (neighbours)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"k-NN evaluation — {len(classes)} NIH classes (epoch {epoch})")
    ax.legend()
    ax.set_xscale("log")
    ax.set_xticks(sorted(args.k))
    ax.set_xticklabels(sorted(args.k))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path("moco_knn.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
