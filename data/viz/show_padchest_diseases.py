"""
Show example chest X-rays for specified PadChest disease labels.
uv run python -m data.viz.show_padchest_diseases
uv run python -m data.viz.show_padchest_diseases --labels "bronchiectasis" "calcified granuloma" "callus rib fracture" --out padchest_examples.png
"""

import argparse
import ast
import csv
import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DEFAULT_LABELS = [
    "bronchiectasis",
    "calcified granuloma",
    "callus rib fracture",
    "hiatal hernia",
    "interstitial pattern",
    "hemidiaphragm elevation",
    "fibrotic band",
    "pulmonary fibrosis",
    "pulmonary mass",
    "reticular interstitial pattern",
    "pneumothorax",
    "tuberculosis",
]


def find_examples(
    data_dir: Path, targets: list[str], n: int
) -> dict[str, list[Path]]:
    """Scan the PadChest CSV and return up to n image paths per target label."""
    disk_index = {p.name: p for p in data_dir.rglob("*.png")}
    print(f"  Images on disk: {len(disk_index)}")

    gz_files = sorted(data_dir.glob("*.csv.gz"))
    csv_files = sorted(data_dir.glob("*.csv"))
    if not gz_files and not csv_files:
        raise FileNotFoundError(f"No CSV found in {data_dir}")

    target_set = {t.lower() for t in targets}
    found: dict[str, list[Path]] = {t: [] for t in target_set}

    def open_csv():
        if gz_files:
            return gzip.open(gz_files[0], "rt")
        return open(csv_files[0])

    with open_csv() as f:
        for row in csv.DictReader(f):
            if row.get("Projection", "").strip() not in ("PA", "AP"):
                continue
            image_id = row.get("ImageID", "").strip()
            if image_id not in disk_index:
                continue
            try:
                labels = ast.literal_eval(row["Labels"])
            except (ValueError, SyntaxError, KeyError):
                continue
            if len(labels) != 1: # only show examples with exactly one label
                continue
            label = str(labels[0]).strip().lower()
            if label in found and len(found[label]) < n:
                found[label].append(disk_index[image_id])
            if all(len(v) >= n for v in found.values()):
                break

    return found


def plot_grid(
    found: dict[str, list[Path]], n: int, out: Path
) -> None:
    labels = sorted(found.keys())
    n_rows = len(labels)
    n_cols = n

    # Compute figsize so axes cells are exactly S×S inches after fixed margins.
    S = 2.2
    left_in, right_in, top_in, bot_in = 1.5, 0.1, 0.35, 0.05
    figw = left_in + n_cols * S + right_in
    figh = top_in  + n_rows * S + bot_in

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figw, figh))
    fig.subplots_adjust(
        left=left_in/figw, right=1-right_in/figw,
        top=1-top_in/figh,  bottom=bot_in/figh,
        wspace=0, hspace=0,
    )
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    for row_idx, label in enumerate(labels):
        paths = found[label]
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if col_idx < len(paths):
                img = np.array(Image.open(paths[col_idx]).resize((224, 224))).astype(np.float32)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                ax.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="auto")
            else:
                ax.text(0.5, 0.5, "not found", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
            ax.axis("off")
            if col_idx == 0:
                ax.text(-0.05, 0.5, label, fontsize=9, transform=ax.transAxes,
                        va="center", ha="right")

    plt.suptitle("PadChest Disease Examples", fontsize=13, fontweight="bold")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show PadChest disease examples in a grid")
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS,
                        help="Disease labels to show")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of examples per label (default: 5)")
    parser.add_argument("--data-dir", type=str, default="datasets/padchest")
    parser.add_argument("--out", type=str, default="padchest_disease_examples.png")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"Searching for {args.n} examples of {len(args.labels)} labels...")
    found = find_examples(data_dir, args.labels, args.n)

    for label in sorted(found.keys()):
        print(f"  {label:<40s} {len(found[label])}/{args.n} found")

    plot_grid(found, args.n, Path(args.out))

