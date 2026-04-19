"""Generate PadChest label-frequency and disease-count charts for the blog.

uv run python -m data.viz.padchest_stats
"""

import csv
import gzip
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_STATIC = Path(__file__).parent.parent.parent.parent / "personal-website/portfolio/static/clear-cxr"
_DATA_DIR = Path("datasets/padchest")

_SELECTED = {
    "bronchiectasis",
    "calcified granuloma",
    "callus rib fracture",
    "fibrotic band",
    "hemidiaphragm elevation",
    "hiatal hernia",
    "interstitial pattern",
    "pulmonary fibrosis",
    "pulmonary mass",
    "reticular interstitial pattern",
}

_RC = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#d0cec8",
    "axes.labelcolor":   "#555555",
    "axes.titlecolor":   "#333333",
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "axes.grid":         True,
    "grid.color":        "#e8e6e0",
    "grid.linewidth":    0.7,
    "grid.linestyle":    "--",
    "xtick.color":       "#888888",
    "ytick.color":       "#888888",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "text.color":        "#555555",
    "savefig.facecolor": "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "lines.linewidth":   1.8,
    "lines.markersize":  5,
    "font.family":       "sans-serif",
}

_BLUE = "#6b8cba"
_ORANGE = "#c47c3e"

_LABEL_RE = re.compile(r"'([^']+)'")


def _parse_labels(raw: str) -> list[str]:
    """Extract label strings from a Python-list-style cell, e.g. ['foo', 'bar']."""
    return _LABEL_RE.findall(raw)


def _count_labels(data_dir: Path, subdir: str | None = None,
                  single_label_only: bool = False) -> Counter:
    gz = sorted(data_dir.glob("*.csv.gz"))
    csv_files = sorted(data_dir.glob("*.csv"))
    counts: Counter = Counter()

    img_filter: set[str] | None = None
    if subdir is not None:
        img_filter = {p.name for p in (data_dir / subdir).rglob("*.png")}

    def open_csv():
        if gz:
            return gzip.open(gz[0], "rt")
        return open(csv_files[0])

    with open_csv() as f:
        for row in csv.DictReader(f):
            if row.get("Projection", "").strip() not in ("PA", "AP"):
                continue
            if img_filter is not None and row.get("ImageID", "").strip() not in img_filter:
                continue
            raw = row.get("Labels", "")
            labels = _parse_labels(raw)
            if single_label_only and len(labels) != 1:
                continue
            for label in labels:
                counts[label.strip().lower()] += 1

    return counts


def plot_single_zip(counts: Counter, subdir: str, total_images: int) -> None:
    """Horizontal bar chart of top labels in one zip, selected diseases in orange."""
    from matplotlib.patches import Patch

    # Top 20 by count, plus any selected diseases not already included
    top_labels = {lbl for lbl, _ in counts.most_common(20)}
    selected_present = {lbl for lbl in _SELECTED if counts.get(lbl, 0) > 0}
    all_labels = sorted(top_labels | selected_present, key=lambda l: -counts.get(l, 0))
    labels = all_labels
    values = [counts.get(lbl, 0) for lbl in labels]
    colors = [_ORANGE if lbl in _SELECTED else _BLUE for lbl in labels]
    alphas = [1.0 if lbl in _SELECTED else 0.55 for lbl in labels]

    with plt.rc_context(_RC):
        n_rows = len(labels)
        fig, ax = plt.subplots(figsize=(6.58, 0.28 * n_rows + 1.0))
        y = np.arange(n_rows)

        for yi, (val, col, alp) in enumerate(zip(values, colors, alphas)):
            ax.barh(yi, val, color=col, alpha=alp, height=0.5, edgecolor="white", linewidth=0.4)

        # Value labels on every bar
        x_max = max(values)
        for yi, val in enumerate(values):
            ax.text(val + x_max * 0.015, yi, str(val), va="center", fontsize=7.0, color="#555555")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("Labeled images", labelpad=6)

        ax.legend(
            handles=[
                Patch(color=_BLUE,   alpha=0.55, label="other findings"),
                Patch(color=_ORANGE, label="selected rare diseases"),
            ],
            loc="lower right", fontsize=9,
            framealpha=0.92, edgecolor="#d0cec8",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.suptitle(
            f"Zip {subdir}: single-label image counts",
            x=0.5, fontsize=11.5, color="#333333", fontfamily="sans-serif",
        )
        out = _STATIC / "padchest_zip_counts.svg"
        fig.savefig(out, format="svg", bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {out}")


def plot_label_frequency(counts: Counter) -> None:
    """Long-tail rank vs count, with selected diseases highlighted."""
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    values_sorted = [cnt for _, cnt in sorted_counts]
    ranks = list(range(1, len(sorted_counts) + 1))

    sel_ranks, sel_values, sel_labels = [], [], []
    for rank, (lbl, cnt) in enumerate(sorted_counts, start=1):
        if lbl in _SELECTED:
            sel_ranks.append(rank)
            sel_values.append(cnt)
            sel_labels.append(lbl)

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.58, 3.46))
        ax.plot(ranks, values_sorted, color=_BLUE, linewidth=1.6, zorder=2)
        ax.scatter(sel_ranks, sel_values, color=_ORANGE, s=40, zorder=4,
                   label="selected diseases")

        for rank, val, lbl in zip(sel_ranks, sel_values, sel_labels):
            ax.annotate(
                lbl,
                xy=(rank, val),
                xytext=(rank + 5, val + max(values_sorted) * 0.04),
                fontsize=7,
                color=_ORANGE,
                arrowprops=dict(arrowstyle="-", color=_ORANGE, lw=0.7),
            )

        ax.set_xlabel("Label rank (sorted by frequency)")
        ax.set_ylabel("Image count (PA/AP)")
        ax.set_title("PadChest: label frequency distribution")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        out = _STATIC / "padchest_label_freq.svg"
        fig.savefig(out, format="svg", bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {out}")


def plot_disease_counts() -> None:
    """Horizontal bar chart of train/val counts for the 10 selected diseases."""
    train_counts = {
        "calcified granuloma":            96,
        "callus rib fracture":            88,
        "interstitial pattern":           93,
        "hemidiaphragm elevation":        81,
        "hiatal hernia":                  81,
        "bronchiectasis":                 40,
        "fibrotic band":                  38,
        "reticular interstitial pattern": 32,
        "pulmonary mass":                 30,
        "pulmonary fibrosis":             24,
    }
    val_counts = {
        "calcified granuloma":            27,
        "callus rib fracture":            21,
        "interstitial pattern":           18,
        "hemidiaphragm elevation":        23,
        "hiatal hernia":                  20,
        "bronchiectasis":                 10,
        "fibrotic band":                  11,
        "reticular interstitial pattern":  8,
        "pulmonary mass":                 11,
        "pulmonary fibrosis":              4,
    }

    diseases = sorted(train_counts, key=lambda d: train_counts[d])
    train_vals = [train_counts[d] for d in diseases]
    val_vals   = [val_counts[d]   for d in diseases]

    y = np.arange(len(diseases))
    bar_h = 0.32

    with plt.rc_context(_RC):
        from matplotlib.patches import Patch

        fig, ax = plt.subplots(figsize=(6.58, 4.0))

        ax.barh(y + bar_h / 2, train_vals, height=bar_h, color=_BLUE,
                edgecolor="white", linewidth=0.4, label="train")
        ax.barh(y - bar_h / 2, val_vals,   height=bar_h, color=_BLUE,
                alpha=0.45, edgecolor="white", linewidth=0.4, label="val")

        x_max = max(train_vals)
        # Value labels — train centered on bar, val nudged down to avoid overlapping row above
        for yi, (tr, vl) in enumerate(zip(train_vals, val_vals)):
            ax.text(tr + x_max * 0.015, yi + bar_h / 2,        str(tr), va="center", fontsize=7.5, color="#555555")
            ax.text(vl + x_max * 0.015, yi - bar_h / 2 - 0.08, str(vl), va="center", fontsize=7.5, color="#888888")

        ax.set_yticks(y)
        ax.set_yticklabels(diseases, fontsize=8.5)
        ax.set_xlabel("Labeled examples", labelpad=6)

        ax.legend(
            handles=[
                Patch(color=_BLUE,            label="train"),
                Patch(color=_BLUE, alpha=0.45, label="val"),
            ],
            loc="lower right", fontsize=9,
            framealpha=0.92, edgecolor="#d0cec8",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.suptitle("Selected disease label counts  (train / val)",
                     x=0.5, fontsize=11.5, color="#333333", fontfamily="sans-serif")
        out = _STATIC / "padchest_disease_counts.svg"
        fig.savefig(out, format="svg", bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    _STATIC.mkdir(parents=True, exist_ok=True)

    subdir = "0"
    print(f"Counting labels in partition {subdir}...")
    zip_counts = _count_labels(_DATA_DIR, subdir=subdir, single_label_only=True)
    total_images = sum(1 for _ in (_DATA_DIR / subdir).rglob("*.png"))
    plot_single_zip(zip_counts, subdir, total_images)

    print("Counting all labels...")
    counts = _count_labels(_DATA_DIR)
    print(f"  {len(counts)} unique labels found")
    plot_label_frequency(counts)
    plot_disease_counts()


if __name__ == "__main__":
    main()
