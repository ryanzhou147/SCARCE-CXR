"""
Generate a 3x5 Grad-CAM comparison across all five initializations.
Rows = hiatal hernia images (low/mid/high ImageNet confidence).
Columns = MoCo v2, BarlowTwins, SparK, ImageNet, Random init.

Usage:
  uv run python -m data.viz.gradcam_comparison \
    --moco-checkpoint outputs_cloud/moco-v2/best.pt \
    --barlow-checkpoint outputs_cloud/barlow/best.pt \
    --spark-checkpoint outputs_cloud/spark/best.pt \
    [--device cpu] [--max-neg 400]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image
from sklearn.linear_model import LogisticRegression

from data.load_backbone import load_raw_backbone
from finetune._data import _TRANSFORM, extract_features, load_negative_pool, load_padchest_splits

DISEASE = "hiatal hernia"
N_ROWS = 3
COL_LABELS = ["MoCo v2", "BarlowTwins", "SparK", "ImageNet", "Random init"]
IN_IDX = 3  # ImageNet column index

_STATIC = (
    Path(__file__).parent.parent.parent.parent
    / "personal-website/portfolio/static/clear-cxr"
)


class _GradCAM:
    """Hooks layer4 of a ResNet backbone for Grad-CAM at the semantic layer."""

    def __init__(self, backbone: nn.Module) -> None:
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        backbone.layer4.register_forward_hook(self._act_hook)
        backbone.layer4.register_full_backward_hook(self._grad_hook)

    def _act_hook(self, module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        self._activations = out

    def _grad_hook(self, module: nn.Module, grad_in: tuple, grad_out: tuple) -> None:
        self._gradients = grad_out[0]

    def compute(
        self,
        backbone: nn.Module,
        coef: np.ndarray,
        intercept: float,
        img: torch.Tensor,
    ) -> np.ndarray:
        coef_t = torch.from_numpy(coef.astype(np.float32)).to(img.device)
        img = img.clone().requires_grad_(True)
        backbone.train(False)
        backbone.zero_grad()
        with torch.enable_grad():
            feats = backbone(img)
            score = feats[0] @ coef_t + intercept
            score.backward()
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def _load_imagenet_backbone(device: torch.device) -> nn.Module:
    m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()
    return m.to(device)


def _load_random_backbone(device: torch.device) -> nn.Module:
    m = tvm.resnet50(weights=None)
    m.fc = nn.Identity()
    return m.to(device)


def _load_image(path: Path) -> tuple[torch.Tensor, np.ndarray]:
    img = Image.open(path).convert("RGB").resize((224, 224))
    orig = np.array(img.convert("L"), dtype=np.float32)
    orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)
    return _TRANSFORM(img).unsqueeze(0), orig


def _fit_probe(
    backbone: nn.Module,
    pos_paths: list[Path],
    neg_paths: list[Path],
    device: torch.device,
) -> tuple[np.ndarray, float]:
    paths = pos_paths + neg_paths
    labels = np.array([1] * len(pos_paths) + [0] * len(neg_paths))
    feats = extract_features(backbone, paths, device)
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    clf.fit(feats, labels)
    return clf.coef_[0], float(clf.intercept_[0])


def _prob(
    backbone: nn.Module,
    coef: np.ndarray,
    intercept: float,
    img: torch.Tensor,
    device: torch.device,
) -> float:
    coef_t = torch.from_numpy(coef.astype(np.float32)).to(device)
    with torch.no_grad():
        score = float(backbone(img.to(device))[0] @ coef_t) + intercept
    return float(torch.sigmoid(torch.tensor(score)))


def _free(backbone: nn.Module) -> None:
    del backbone
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moco-checkpoint", default="outputs_cloud/moco-v2/best.pt")
    parser.add_argument("--barlow-checkpoint", default="outputs_cloud/barlow/best.pt")
    parser.add_argument("--spark-checkpoint", default="outputs_cloud/spark/best.pt")
    parser.add_argument("--data-dir", default="datasets/padchest")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-neg", type=int, default=400,
                        help="Cap negative pool size (smaller = faster on CPU)")
    parser.add_argument("--device", default=None,
                        help="Force device: 'cpu' or 'cuda' (default: auto-detect)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data splits...")
    train_by_label, val_by_label = load_padchest_splits(
        Path(args.data_dir), args.val_frac, args.seed
    )
    pos_train = train_by_label[DISEASE]
    pos_val = val_by_label[DISEASE]
    neg_train_full, _ = load_negative_pool(
        Path(args.data_dir), set(train_by_label.keys()), args.val_frac
    )
    rng = np.random.default_rng(args.seed)
    neg_train = list(rng.choice(neg_train_full, size=min(args.max_neg, len(neg_train_full)), replace=False))

    # Backbone specs: (label, loader_fn_or_ckpt_path)
    backbone_specs = [
        ("MoCo v2",     args.moco_checkpoint),
        ("BarlowTwins", args.barlow_checkpoint),
        ("SparK",       args.spark_checkpoint),
        ("ImageNet",    "imagenet"),
        ("Random init", "random"),
    ]

    # Pass 1: fit probes and score all val images per backbone (one at a time)
    print("Pass 1: fitting probes and scoring val images...")
    all_val_probs: list[list[float]] = []  # [backbone_idx][image_idx]
    probes: list[tuple[np.ndarray, float]] = []

    for label, spec in backbone_specs:
        print(f"  {label}...")
        if spec == "imagenet":
            backbone = _load_imagenet_backbone(device)
        elif spec == "random":
            backbone = _load_random_backbone(device)
        else:
            ckpt = torch.load(spec, map_location=device, weights_only=False)
            backbone, _ = load_raw_backbone(ckpt, device)

        coef, intercept = _fit_probe(backbone, pos_train, neg_train, device)
        probes.append((coef, intercept))

        probs = []
        for path in pos_val:
            img_tensor, _ = _load_image(path)
            probs.append(_prob(backbone, coef, intercept, img_tensor, device))
        all_val_probs.append(probs)
        _free(backbone)

    # Pick 3 evenly spaced val images
    chosen_idxs = list(np.linspace(0, len(pos_val) - 1, N_ROWS, dtype=int))
    chosen_paths = [pos_val[i] for i in chosen_idxs]
    in_probs_chosen = [all_val_probs[IN_IDX][i] for i in chosen_idxs]
    print(f"  Chosen image indices: {chosen_idxs}")

    # Pass 2: compute GradCAM for the 3 chosen images (one backbone at a time)
    print("Pass 2: computing Grad-CAM...")
    n_cols = len(backbone_specs)
    cams: list[list[np.ndarray | None]] = [[None] * n_cols for _ in range(N_ROWS)]
    probs_grid: list[list[float]] = [[0.0] * n_cols for _ in range(N_ROWS)]
    originals: list[np.ndarray | None] = [None] * N_ROWS

    for row, path in enumerate(chosen_paths):
        _, orig = _load_image(path)
        originals[row] = orig

    for col, (label, spec) in enumerate(backbone_specs):
        print(f"  {label}...")
        if spec == "imagenet":
            backbone = _load_imagenet_backbone(device)
        elif spec == "random":
            backbone = _load_random_backbone(device)
        else:
            ckpt = torch.load(spec, map_location=device, weights_only=False)
            backbone, _ = load_raw_backbone(ckpt, device)

        gcam = _GradCAM(backbone)
        coef, intercept = probes[col]

        for row, path in enumerate(chosen_paths):
            img_tensor, _ = _load_image(path)
            cams[row][col] = gcam.compute(backbone, coef, intercept, img_tensor.to(device))
            probs_grid[row][col] = _prob(backbone, coef, intercept, img_tensor, device)
        _free(backbone)

    # Plot
    fig, axes = plt.subplots(N_ROWS, n_cols, figsize=(n_cols * 3.0, N_ROWS * 3.2))

    for col, (label, _) in enumerate(backbone_specs):
        axes[0, col].set_title(label, fontsize=13, fontweight="bold")

    for row in range(N_ROWS):
        axes[row, 0].set_ylabel(f"image {row + 1}", fontsize=10)
        for col in range(n_cols):
            ax = axes[row, col]
            ax.imshow(originals[row], cmap="gray", vmin=0, vmax=1)
            ax.imshow(cams[row][col], cmap="jet", alpha=0.45, vmin=0, vmax=1)
            ax.set_xlabel(f"p={probs_grid[row][col]:.2f}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "Grad-CAM: Hiatal Hernia, MoCo v2, BarlowTwins, SparK, ImageNet, Random (layer4)",
        fontsize=15,
        fontweight="bold",
        color="#333333",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = Path(args.out) if args.out else _STATIC / "gradcam_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.close()


if __name__ == "__main__":
    main()
