"""Frozen backbone + logistic regression probe for per-disease binary classification on PadChest.
Run this first to compare checkpoints quickly before committing to full fine-tuning.

Usage:
  python -m finetune.probe --checkpoint outputs/moco-v3/best.pt
  python -m finetune.probe --checkpoint outputs/moco-v3/best.pt --n 1 5 10 20 all
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score, roc_auc_score

from data.load_backbone import (
    load_feature_extractor,
    load_imagenet_feature_extractor,
    method_name,
)
from finetune._data import (
    FEW_SHOT_NS,
    N_TRIALS,
    extract_features,
    load_negative_pool,
    load_padchest_splits,
)
from finetune._plots import plot_mean_auc, plot_per_disease


def run_binary_probe(
    pos_train: np.ndarray,
    neg_train: np.ndarray,
    pos_val: np.ndarray,
    neg_val: np.ndarray,
    n_shot: int,
    rng: random.Random,
) -> tuple[float, float, float]:
    n_pos = len(pos_train) if n_shot == -1 else min(n_shot, len(pos_train))
    pos_idx = (
        list(range(len(pos_train)))
        if n_shot == -1
        else rng.sample(range(len(pos_train)), n_pos)
    )
    neg_idx = rng.sample(range(len(neg_train)), min(n_pos, len(neg_train)))

    X = np.concatenate([pos_train[pos_idx], neg_train[neg_idx]])
    y = np.array([1] * len(pos_idx) + [0] * len(neg_idx))

    if n_shot == 1:
        # CV requires at least 2 samples per class
        clf = LogisticRegression(max_iter=1000, C=1.0)
    else:
        clf = LogisticRegressionCV(
            Cs=[0.01, 0.1, 1.0, 10.0, 100.0],
            max_iter=1000,
            cv=min(5, n_pos),
            refit=True,
        )
    clf.fit(X, y)

    val_neg_idx = rng.sample(range(len(neg_val)), min(len(pos_val), len(neg_val)))
    X_val = np.concatenate([pos_val, neg_val[val_neg_idx]])
    y_val = np.array([1] * len(pos_val) + [0] * len(val_neg_idx))
    preds = clf.predict(X_val)
    proba = clf.predict_proba(X_val)[:, 1]

    acc = float((preds == y_val).mean())
    f1 = float(f1_score(y_val, preds, zero_division=0))
    try:
        auc = float(roc_auc_score(y_val, proba))
    except ValueError:
        auc = float("nan")
    return acc, f1, auc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/moco/best.pt")
    parser.add_argument("--data-dir", type=str, default="datasets/padchest")
    parser.add_argument(
        "--n",
        type=lambda v: -1 if v == "all" else int(v),  # -1 = use all available training examples
        nargs="+",
        default=FEW_SHOT_NS,
    )
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classes", type=str, nargs="+", default=None)
    parser.add_argument("--max-neg-train", type=int, default=5000)
    parser.add_argument("--max-neg-val", type=int, default=2000)
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

    # Concatenate all diseases into one array so we do a single forward pass per init,
    # then slice by label index per disease inside the loop.
    label_to_idx    = {c: i for i, c in enumerate(classes)}
    all_train_paths = [p for c in classes for p in train_by_label[c]]
    all_train_labels = np.array([label_to_idx[c] for c in classes for _ in train_by_label[c]])
    all_val_paths   = [p for c in classes for p in val_by_label[c]]
    all_val_labels  = np.array([label_to_idx[c] for c in classes for _ in val_by_label[c]])

    INITS = [
        ("SSL pretrained", False, False),
        ("ImageNet",       False, True),
        ("Random init",    True,  False),
    ]

    all_results: dict = {}  # [init_name][disease][n] = {"auc": (mean, std), "_raw": ...}

    for init_name, is_random, is_imagenet in INITS:
        print(f"\n{'─' * 60}")
        print(f"Init: {init_name}")
        print(f"{'─' * 60}")
        all_results[init_name] = {}

        if is_imagenet:
            backbone = load_imagenet_feature_extractor(device)
        else:
            backbone = load_feature_extractor(checkpoint, device, random_init=is_random)

        rare_tr_feats  = extract_features(backbone, all_train_paths, device)
        rare_val_feats = extract_features(backbone, all_val_paths, device)
        neg_tr_feats   = extract_features(backbone, neg_train_paths, device)
        neg_val_feats  = extract_features(backbone, neg_val_paths, device)

        for disease in classes:
            d_idx = label_to_idx[disease]
            pos_tr  = rare_tr_feats[all_train_labels == d_idx]
            pos_val = rare_val_feats[all_val_labels == d_idx]
            results_n: dict[int, dict] = {}

            for n in sorted(args.n):
                trial_metrics: dict[str, list] = {"acc": [], "f1": [], "auc": []}
                for t in range(N_TRIALS):
                    trial_rng = random.Random(args.seed + t)
                    acc, f1, auc = run_binary_probe(
                        pos_tr, neg_tr_feats, pos_val, neg_val_feats, n, trial_rng
                    )
                    trial_metrics["acc"].append(acc)
                    trial_metrics["f1"].append(f1)
                    trial_metrics["auc"].append(auc)

                summary = {k: (float(np.mean(v)), float(np.std(v))) for k, v in trial_metrics.items()}
                results_n[n] = {**summary, "_raw": {k: list(v) for k, v in trial_metrics.items()}}

            all_results[init_name][disease] = results_n

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

    print(f"\nSignificance tests — probe (paired t-test on AUC, {N_TRIALS} trials)")
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
    out  = Path(f"{mname}_ep{epoch}_probe_binary.png")
    out2 = Path(f"{mname}_ep{epoch}_probe_per_disease.png")
    plot_mean_auc(all_results, classes, ns, mname, epoch, out, title_prefix="Probe")
    plot_per_disease(all_results, classes, ns, mname, epoch, out2, title_prefix="Probe")


if __name__ == "__main__":
    main()
