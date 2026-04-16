"""
Usage:
  python -m data.viz.layer_viz --checkpoint outputs/moco-v3/best.pt --disease "hiatal hernia"
  python -m data.viz.layer_viz --checkpoint outputs/moco-v3/best.pt --disease "bronchiectasis" --n-images 4
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

from data.load_backbone import load_raw_backbone, method_name
from finetune._data import (
    _TRANSFORM,
    extract_features,
    load_negative_pool,
    load_padchest_splits,
)

METHOD_DISPLAY = {"moco": "MoCo v2", "dino": "DINO", "barlow": "BarlowTwins", "spark": "SparK"}

LAYERS = ["layer1", "layer2", "layer3", "layer4"]
LAYER_LABELS = {
    "layer1": "layer1\nedges/contrast",
    "layer2": "layer2\ntextures/patterns",
    "layer3": "layer3\nshapes/regions",
    "layer4": "layer4\nsemantic (disease)",
}


class _MultiLayerGradCAM:
    def __init__(self, backbone: nn.Module):
        self.activations: dict[str, torch.Tensor] = {}
        self.gradients: dict[str, torch.Tensor] = {}
        for name in LAYERS:
            layer = getattr(backbone, name)
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

    def compute(self, backbone, coef, intercept, img_tensor) -> dict[str, np.ndarray]:
        coef_t = torch.from_numpy(coef.astype(np.float32)).to(img_tensor.device)
        img_tensor = img_tensor.clone().requires_grad_(True)
        backbone.train(False)
        backbone.zero_grad()

        with torch.enable_grad():
            feats = backbone(img_tensor)
            score = feats[0] @ coef_t + intercept
            score.backward()

        cams = {}
        for name in LAYERS:
            weights = self.gradients[name].mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * self.activations[name]).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
            cam = cam.squeeze().detach().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cams[name] = cam
        return cams


def _load_image(path: Path) -> tuple[torch.Tensor, np.ndarray]:
    img = Image.open(path).convert("RGB").resize((224, 224))
    original = np.array(img.convert("L"), dtype=np.float32)
    original = (original - original.min()) / (original.max() - original.min() + 1e-8)
    return _TRANSFORM(img).unsqueeze(0), original


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/moco-v3/best.pt")
    parser.add_argument("--disease", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="datasets/padchest")
    parser.add_argument("--n-images", type=int, default=4)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    disease = args.disease.lower()

    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mname = method_name(checkpoint)
    epoch = checkpoint["epoch"] + 1
    print(f"  {mname} ep{epoch}")

    backbone, _ = load_raw_backbone(checkpoint, device)

    print("Loading splits...")
    train_by_label, val_by_label = load_padchest_splits(data_dir, args.val_frac, args.seed)
    if disease not in train_by_label:
        raise ValueError(f"'{disease}' not found. Available: {sorted(train_by_label.keys())}")

    pos_train = train_by_label[disease]
    pos_val = val_by_label[disease]
    neg_train, _ = load_negative_pool(data_dir, set(train_by_label.keys()), args.val_frac)

    print("Fitting probe...")
    coef, intercept = _fit_probe(backbone, pos_train, neg_train, device)

    gcam = _MultiLayerGradCAM(backbone)

    vis_paths = pos_val[: args.n_images]
    n_rows = len(vis_paths)

    fig, axes = plt.subplots(n_rows, 5, figsize=(15, n_rows * 3))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for j, title in enumerate(["original"] + [LAYER_LABELS[l] for l in LAYERS]):
        axes[0, j].set_title(title, fontsize=11)

    for i, path in enumerate(vis_paths):
        img_tensor, original = _load_image(path)
        img_tensor = img_tensor.to(device)
        cams = gcam.compute(backbone, coef, intercept, img_tensor)
        prob = _pred_prob(backbone, coef, intercept, img_tensor.detach())

        axes[i, 0].imshow(original, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_ylabel(f"p={prob:.2f}", fontsize=11, rotation=0, labelpad=30)
        axes[i, 0].axis("off")

        for j, name in enumerate(LAYERS):
            axes[i, j + 1].imshow(original, cmap="gray", vmin=0, vmax=1)
            axes[i, j + 1].imshow(cams[name], cmap="jet", alpha=0.45, vmin=0, vmax=1)
            axes[i, j + 1].axis("off")

    display_name = METHOD_DISPLAY.get(mname, mname)
    fig.suptitle(f"Layer-by-layer Grad-CAM: {disease} ({display_name})", fontsize=18, fontweight="bold", color="#333333")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    _STATIC = Path(__file__).parent.parent.parent.parent / "personal-website/portfolio/static/clear-cxr"
    out = (
        Path(args.out)
        if args.out
        else _STATIC / f"layer_gradcam_{mname.replace(' ', '_').lower()}.png"
    )
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.close()


if __name__ == "__main__":
    main()
