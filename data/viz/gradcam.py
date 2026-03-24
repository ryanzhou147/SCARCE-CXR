"""
Usage:
  python -m data.viz.gradcam --checkpoint outputs/moco-v3/best.pt --disease "hiatal hernia"
  python -m data.viz.gradcam --checkpoint outputs/moco-v3/best.pt --disease "hiatal hernia" --compare
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression

from data.load_backbone import load_imagenet_raw_backbone, load_raw_backbone, method_name
from finetune._data import (
    _TRANSFORM,
    extract_features,
    load_negative_pool,
    load_padchest_splits,
)


class _GradCAM:
    def __init__(self, layer4: nn.Module):
        self.activations = None
        self.gradients = None
        layer4.register_forward_hook(self._save_activations)
        layer4.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def compute(self, backbone, coef, intercept, img_tensor):
        coef_t = torch.from_numpy(coef.astype(np.float32)).to(img_tensor.device)
        img_tensor = img_tensor.clone().requires_grad_(True)
        backbone.train(False)
        backbone.zero_grad()

        with torch.enable_grad():
            feats = backbone(img_tensor)
            score = feats[0] @ coef_t + intercept
            score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def _load_image(path: Path) -> tuple[torch.Tensor, np.ndarray]:
    img = Image.open(path).convert("RGB").resize((224, 224))
    original = np.array(img.convert("L"), dtype=np.float32)
    original = (original - original.min()) / (original.max() - original.min() + 1e-8)
    return _TRANSFORM(img).unsqueeze(0), original


def _overlay(ax, original: np.ndarray, cam: np.ndarray, title: str):
    ax.imshow(original, cmap="gray", vmin=0, vmax=1)
    ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _fit_probe(backbone, pos_train, neg_train, device):
    all_paths = pos_train + neg_train
    all_labels = np.array([1] * len(pos_train) + [0] * len(neg_train))
    feats = extract_features(backbone, all_paths, device)
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(feats, all_labels)
    return clf.coef_[0], float(clf.intercept_[0])


@torch.no_grad()
def _pred_prob(backbone, coef, intercept, img_tensor):
    coef_t = torch.from_numpy(coef.astype(np.float32)).to(img_tensor.device)
    score = float(backbone(img_tensor)[0] @ coef_t) + intercept
    return float(torch.sigmoid(torch.tensor(score)))


def _run_comparison(args, device, data_dir, disease):
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ssl_label = f"{method_name(checkpoint)} ep{checkpoint['epoch'] + 1}"

    print("Loading backbones...")
    ssl_backbone, _ = load_raw_backbone(checkpoint, device)
    imagenet_backbone, _ = load_imagenet_raw_backbone(device)
    random_backbone, _ = load_raw_backbone(checkpoint, device, random_init=True)

    print("Loading splits...")
    train_by_label, val_by_label = load_padchest_splits(data_dir, args.val_frac, args.seed)
    if disease not in train_by_label:
        raise ValueError(f"'{disease}' not found. Available: {sorted(train_by_label.keys())}")

    pos_train = train_by_label[disease]
    pos_val = val_by_label[disease]
    neg_train, _ = load_negative_pool(data_dir, set(train_by_label.keys()), args.val_frac)

    backbones = [
        (ssl_backbone, ssl_label),
        (imagenet_backbone, "imagenet"),
        (random_backbone, "random"),
    ]

    print("Fitting probes...")
    probes, gcams = [], []
    for backbone, label in backbones:
        print(f"  {label}...")
        probes.append(_fit_probe(backbone, pos_train, neg_train, device))
        gcams.append(_GradCAM(backbone.layer4))

    vis_paths = pos_val[: args.n_images]
    n_images = len(vis_paths)
    col_labels = [label for _, label in backbones]

    fig, axes = plt.subplots(n_images, 3, figsize=(9, n_images * 3))
    if n_images == 1:
        axes = axes.reshape(1, -1)

    for row, path in enumerate(vis_paths):
        img_tensor, original = _load_image(path)
        img_tensor = img_tensor.to(device)
        for col, ((backbone, _), (coef, intercept), gcam) in enumerate(
            zip(backbones, probes, gcams)
        ):
            cam = gcam.compute(backbone, coef, intercept, img_tensor)
            prob = _pred_prob(backbone, coef, intercept, img_tensor.detach())
            header = f"{col_labels[col]}\n" if row == 0 else ""
            _overlay(axes[row, col], original, cam, f"{header}p={prob:.2f}")

    fig.suptitle(f"Grad-CAM Comparison — {disease}", fontsize=12)
    plt.tight_layout()
    out = Path(f"comparison_gradcam_{disease.replace(' ', '_')}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/moco-v3/best.pt")
    parser.add_argument("--disease", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="datasets/padchest")
    parser.add_argument("--n-images", type=int, default=6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--init", choices=["ssl", "imagenet", "random"], default="ssl")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    disease = args.disease.lower()

    if args.compare:
        _run_comparison(args, device, data_dir, disease)
        return

    print(f"Loading backbone ({args.init})...")
    if args.init == "imagenet":
        backbone, _ = load_imagenet_raw_backbone(device)
        mname, epoch = "imagenet", 0
    else:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        mname = method_name(checkpoint)
        epoch = checkpoint["epoch"] + 1
        random_init = args.init == "random"
        backbone, _ = load_raw_backbone(checkpoint, device, random_init=random_init)
        if random_init:
            mname = "random"
    print(f"  {mname} ep{epoch}")

    print("Loading splits...")
    train_by_label, val_by_label = load_padchest_splits(data_dir, args.val_frac, args.seed)
    if disease not in train_by_label:
        raise ValueError(f"'{disease}' not found. Available: {sorted(train_by_label.keys())}")

    pos_train = train_by_label[disease]
    pos_val = val_by_label[disease]
    neg_train, _ = load_negative_pool(data_dir, set(train_by_label.keys()), args.val_frac)

    coef, intercept = _fit_probe(backbone, pos_train, neg_train, device)
    gcam = _GradCAM(backbone.layer4)

    vis_paths = pos_val[: args.n_images]
    n_cols = min(3, len(vis_paths))
    n_rows = (len(vis_paths) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = np.array(axes).reshape(-1)

    for i, path in enumerate(vis_paths):
        img_tensor, original = _load_image(path)
        img_tensor = img_tensor.to(device)
        cam = gcam.compute(backbone, coef, intercept, img_tensor)
        prob = _pred_prob(backbone, coef, intercept, img_tensor.detach())
        _overlay(axes[i], original, cam, f"p={prob:.2f}")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    label = mname if mname in ("imagenet", "random") else f"{mname} ep{epoch}"
    fig.suptitle(f"Grad-CAM — {disease}\n{label}", fontsize=11)
    plt.tight_layout()

    out = (
        Path(args.out)
        if args.out
        else Path(f"{label.replace(' ', '_')}_gradcam_{disease.replace(' ', '_')}.png")
    )
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.close()


if __name__ == "__main__":
    main()
