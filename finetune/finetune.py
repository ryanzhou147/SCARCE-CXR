"""Binary few-shot gradient fine-tuning on PadChest rare diseases.

Contrastive (MoCo, BarlowTwins, DINO): unfreeze layer2 onwards.
Restorative (SparK): unfreeze layer3 onwards (layer2 adapted during pretraining).

Usage:
  python -m finetune.finetune --checkpoint outputs/moco-v3/best.pt
  python -m finetune.finetune --checkpoint outputs/moco-v3/best.pt --n 5 20 all
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data.load_backbone import (
    load_imagenet_raw_backbone,
    load_raw_backbone,
    method_name,
    unfreeze_for_finetuning,
)
from finetune._data import (
    FEW_SHOT_NS,
    N_TRIALS,
    _TRANSFORM,
    _XRAY_MEAN,
    _XRAY_STD,
    extract_features,
    load_negative_pool,
    load_padchest_splits,
)
from finetune._plots import plot_mean_auc, plot_per_disease

_FINETUNE_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=_XRAY_MEAN, std=_XRAY_STD),
    # ColorJitter and GaussianBlur run ~6x faster on tensors than PIL images.
    transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8)], p=0.8),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=9)], p=0.5),
])


class LabeledImageDataset(Dataset):
    def __init__(self, paths: list[Path], labels: np.ndarray, transform):
        # Pre-load into memory — avoids disk reads every epoch for small few-shot sets.
        self.images = [Image.open(p).convert("RGB") for p in paths]
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.transform(self.images[idx]), int(self.labels[idx])


def _prototype_head(
    shot_feats: np.ndarray, shot_labels: np.ndarray, n_classes: int
) -> nn.Linear:
    """Linear head initialized to per-class mean L2-normalized feature vectors.

    Prototype init gives the head a reasonable starting point with only 5–20 examples —
    random init would need far more gradient steps to learn the same class separation.
    """
    feat_dim = shot_feats.shape[1]
    protos = np.zeros((n_classes, feat_dim), dtype=np.float32)
    counts = np.zeros(n_classes, dtype=np.int32)
    for feat, label in zip(shot_feats, shot_labels):
        protos[label] += feat
        counts[label] += 1
    counts = np.maximum(counts, 1)
    protos /= counts[:, None]
    protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8

    head = nn.Linear(feat_dim, n_classes, bias=True)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(protos))
        head.bias.zero_()
    return head


def run_binary_finetune(
    backbone: nn.Module,
    method: str,
    pos_train_paths: list[Path],
    neg_train_paths: list[Path],
    pos_val_paths: list[Path],
    neg_val_paths: list[Path],
    n_shot: int,
    rng: random.Random,
    device: torch.device,
    n_epochs: int = 100,
    lr_head: float = 1e-3,
    lr_backbone: float = 1e-4,
    wd: float = 0.01,
) -> tuple[float, float, float]:
    n_pos = len(pos_train_paths) if n_shot == -1 else min(n_shot, len(pos_train_paths))
    pos_idx = (
        list(range(len(pos_train_paths)))
        if n_shot == -1
        else rng.sample(range(len(pos_train_paths)), n_pos)
    )
    neg_idx = rng.sample(range(len(neg_train_paths)), min(n_pos, len(neg_train_paths)))
    shot_paths  = [pos_train_paths[i] for i in pos_idx] + [neg_train_paths[i] for i in neg_idx]
    shot_labels = np.array([1] * len(pos_idx) + [0] * len(neg_idx))

    backbone.train(False)
    shot_feats = extract_features(backbone, shot_paths, device)
    head = _prototype_head(shot_feats, shot_labels, 2).to(device)

    saved_state = {k: v.clone() for k, v in backbone.state_dict().items()}
    unfreeze_for_finetuning(backbone, method)

    bb_params = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": bb_params,         "lr": lr_backbone},
            {"params": head.parameters(), "lr": lr_head},
        ],
        weight_decay=wd,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        LabeledImageDataset(shot_paths, shot_labels, _FINETUNE_TRANSFORM),
        batch_size=min(32, len(shot_paths)),
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    for _ in range(n_epochs):
        backbone.train()
        # Keep BatchNorm frozen to preserve pretraining statistics.
        for m in backbone.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.train(False)
        head.train()
        for imgs, labels in loader:
            feats = F.normalize(backbone(imgs.to(device)), dim=1)
            loss  = criterion(head(feats), labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    backbone.train(False)
    head.train(False)

    n_val_neg = min(len(pos_val_paths), len(neg_val_paths))
    sampled_neg_val = rng.sample(neg_val_paths, n_val_neg)
    val_paths_all  = pos_val_paths + sampled_neg_val
    val_labels_all = np.array([1] * len(pos_val_paths) + [0] * n_val_neg)

    val_feats = extract_features(backbone, val_paths_all, device)
    with torch.no_grad():
        logits = head(torch.from_numpy(val_feats).to(device))
        preds  = logits.argmax(1).cpu().numpy()
        proba  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    acc = float((preds == val_labels_all).mean())
    f1  = float(f1_score(val_labels_all, preds, zero_division=0))
    try:
        auc = float(roc_auc_score(val_labels_all, proba))
    except ValueError:
        auc = float("nan")

    backbone.load_state_dict(saved_state)
    for p in backbone.parameters():
        p.requires_grad_(False)

    return acc, f1, auc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/moco/best.pt")
    parser.add_argument("--data-dir", type=str, default="datasets/padchest")
    parser.add_argument(
        "--n",
        type=lambda v: -1 if v == "all" else int(v),
        nargs="+",
        default=FEW_SHOT_NS,
    )
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classes", type=str, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--max-neg-train", type=int, default=5000)
    parser.add_argument("--max-neg-val", type=int, default=2000)
    parser.add_argument("--wandb-project", type=str, default="cmvr-finetune")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    epoch = checkpoint["epoch"] + 1
    mname = method_name(checkpoint)
    print(f"  Method: {mname}  epoch: {epoch}  device: {device}")

    print("Loading PadChest rare disease splits...")
    train_by_label, val_by_label = load_padchest_splits(data_dir, args.val_frac, args.seed)
    if args.classes:
        keep = set(args.classes)
        train_by_label = {c: v for c, v in train_by_label.items() if c in keep}
        val_by_label   = {c: v for c, v in val_by_label.items() if c in keep}
    classes = sorted(train_by_label)
    print(f"Rare disease classes ({len(classes)}): {classes}")

    print("Loading negative pool...")
    neg_train_paths, neg_val_paths = load_negative_pool(
        data_dir, set(classes), args.val_frac, args.max_neg_train, args.max_neg_val, args.seed
    )

    # Keyed by method+epoch so different checkpoints don't clobber each other.
    state_path = Path(f"{mname}_ep{epoch}_finetune_state.json")
    all_results: dict = {}
    wandb_run_id = None

    if state_path.exists():
        state = json.loads(state_path.read_text())
        wandb_run_id = state.get("wandb_run_id")
        for init_name, diseases in state.get("results", {}).items():
            all_results[init_name] = {
                disease: {int(k): v for k, v in ns.items()}
                for disease, ns in diseases.items()
            }
        n_done = sum(len(ns) for d in all_results.values() for ns in d.values())
        print(f"Resuming from {state_path} ({n_done} disease×n pairs complete)")

    def _save() -> None:
        serializable = {
            init: {d: {str(n): v for n, v in ns.items()} for d, ns in ds.items()}
            for init, ds in all_results.items()
        }
        state_path.write_text(json.dumps({"wandb_run_id": wandb_run_id, "results": serializable}, indent=2))

    if not args.no_wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"{mname}_ep{epoch}_finetune",
            id=wandb_run_id,
            resume="allow",
            config={
                "method": mname,
                "epoch": epoch,
                "n_shots": args.n,
                "n_trials": N_TRIALS,
                "epochs": args.epochs,
                "lr_head": args.lr_head,
                "lr_backbone": args.lr_backbone,
                "classes": classes,
            },
        )
        wandb_run_id = run.id
        _save()

    INITS = [
        ("SSL pretrained", False, False),
        ("ImageNet",       False, True),
        ("Random init",    True,  False),
    ]

    for init_name, is_random, is_imagenet in INITS:
        print(f"\n{'─' * 60}")
        print(f"Init: {init_name}")
        print(f"{'─' * 60}")
        all_results.setdefault(init_name, {})

        backbone = None  # loaded lazily — skipped entirely if all results already cached

        for disease in classes:
            all_results[init_name].setdefault(disease, {})
            pos_tr_paths  = train_by_label[disease]
            pos_val_paths = val_by_label[disease]

            for n in sorted(args.n):
                if n in all_results[init_name][disease]:
                    print(f"  skip  {disease}  n={n}")
                    continue

                if backbone is None:
                    if is_imagenet:
                        backbone, bb_method = load_imagenet_raw_backbone(device)
                    else:
                        backbone, bb_method = load_raw_backbone(checkpoint, device, random_init=is_random)

                trial_metrics: dict[str, list] = {"acc": [], "f1": [], "auc": []}
                for t in range(N_TRIALS):
                    trial_rng = random.Random(args.seed + t)
                    acc, f1, auc = run_binary_finetune(
                        backbone, bb_method,
                        pos_tr_paths, neg_train_paths,
                        pos_val_paths, neg_val_paths,
                        n, trial_rng, device,
                        n_epochs=args.epochs,
                        lr_head=args.lr_head,
                        lr_backbone=args.lr_backbone,
                    )
                    trial_metrics["acc"].append(acc)
                    trial_metrics["f1"].append(f1)
                    trial_metrics["auc"].append(auc)
                    if not args.no_wandb:
                        wandb.log({
                            f"trial/{init_name}/{disease}/{n}shot/auc": auc,
                            f"trial/{init_name}/{disease}/{n}shot/f1": f1,
                        })

                summary = {k: (float(np.mean(v)), float(np.std(v))) for k, v in trial_metrics.items()}
                all_results[init_name][disease][n] = {
                    **summary,
                    "_raw": {k: list(v) for k, v in trial_metrics.items()},
                }
                _save()

                if not args.no_wandb:
                    wandb.log({
                        f"mean/{init_name}/{disease}/{n}shot/auc": summary["auc"][0],
                        f"mean/{init_name}/{disease}/{n}shot/f1": summary["f1"][0],
                    })

        shot_labels = ["all" if n == -1 else str(n) for n in sorted(args.n)]
        print(f"\n  {'Disease':<42s}", end="")
        for lbl in shot_labels:
            print(f"  {lbl + '-shot AUC':>14s}", end="")
        print()
        for disease in classes:
            print(f"  {disease:<42s}", end="")
            for n in sorted(args.n):
                m, s = all_results[init_name][disease][n]["auc"]
                print(f"  {m:.3f}±{s:.3f}      ", end="")
            print()

    print(f"\nSignificance tests — paired t-test on AUC, {N_TRIALS} trials")
    for other in ("ImageNet", "Random init"):
        if other not in all_results:
            continue
        for n in sorted(args.n):
            ssl_aucs   = [np.mean(all_results["SSL pretrained"][d][n]["_raw"]["auc"]) for d in classes]
            other_aucs = [np.mean(all_results[other][d][n]["_raw"]["auc"]) for d in classes]
            t, p = stats.ttest_rel(ssl_aucs, other_aucs)
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
            lbl = "all" if n == -1 else f"{n:3d}"
            print(f"  {lbl}-shot  SSL vs {other:<15s}: t={t:+.2f}  p={p:.3f}  {sig}")

    ns = [n for n in sorted(args.n) if n != -1]
    out  = Path(f"{mname}_ep{epoch}_finetune_binary.png")
    out2 = Path(f"{mname}_ep{epoch}_finetune_per_disease.png")
    plot_mean_auc(all_results, classes, ns, mname, epoch, out, title_prefix="Finetune")
    plot_per_disease(all_results, classes, ns, mname, epoch, out2, title_prefix="Finetune")

    if not args.no_wandb:
        cols = ["init", "disease"] + [f"{n}-shot AUC" for n in sorted(args.n) if n != -1]
        table = wandb.Table(columns=cols)
        for init_name in all_results:
            for disease in classes:
                row = [init_name, disease]
                for n in sorted(args.n):
                    if n == -1:
                        continue
                    m, _ = all_results[init_name][disease][n]["auc"]
                    row.append(round(m, 3))
                table.add_data(*row)
        wandb.log({"results_table": table, "plot": wandb.Image(str(out)), "plot_per_disease": wandb.Image(str(out2))})
        wandb.finish()


if __name__ == "__main__":
    main()
