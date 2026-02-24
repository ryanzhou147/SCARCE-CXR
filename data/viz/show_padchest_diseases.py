"""Show example chest X-rays for specified PadChest disease labels.

Finds N examples of each requested label and displays them in a grid.
Each row = one disease, each column = one example image.

Usage:
  uv run python -m data.show_padchest_diseases
  uv run python -m data.show_padchest_diseases --labels "copd signs" nodule scoliosis
  uv run python -m data.show_padchest_diseases --n 3 --out padchest_examples.png
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
            import gzip as gz_mod
            return gz_mod.open(gz_files[0], "rt")
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
            if len(labels) != 1:
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

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.8))
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    for row_idx, label in enumerate(labels):
        paths = found[label]
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if col_idx < len(paths):
                img = np.array(Image.open(paths[col_idx])).astype(np.float32)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                ax.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="auto")
            else:
                ax.text(0.5, 0.5, "not found", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=9, rotation=0, labelpad=80,
                              va="center", ha="right")

    plt.suptitle("PadChest Disease Examples", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out}")


def main() -> None:
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


if __name__ == "__main__":
    main()
