"""
Visualise the domain gap between NIH and PadChest datasets.
uv run python -m data.viz.domain_gap --nih-dir datasets/nih-chest-xrays --padchest-dir datasets/padchest
"""

import csv
import gzip
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from data.dataloader import collect_image_paths


def _load_gray(path: Path, size: int = 224) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def _collect_padchest_pa(root: Path) -> list[Path]:
    """Return PA/AP-only PadChest image paths using the CSV projection column."""
    root = Path(root)
    gz_candidates = sorted(root.glob("*.csv.gz"))
    csv_candidates = sorted(root.glob("*.csv"))

    if not gz_candidates and not csv_candidates:
        print("  WARNING: no PadChest CSV found — including all views (may include laterals)")
        return sorted(root.rglob("*.png"))

    disk_index = {p.name: p for p in root.rglob("*.png")}

    def open_csv():
        if gz_candidates:
            return gzip.open(gz_candidates[0], "rt", encoding="utf-8", errors="replace")
        return open(csv_candidates[0], encoding="utf-8", errors="replace")

    pa_paths = []
    with open_csv() as f:
        for row in csv.DictReader(f):
            if row.get("Projection", "").strip() not in ("PA", "AP"):
                continue
            image_id = row.get("ImageID", "").strip()
            if image_id and image_id in disk_index:
                pa_paths.append(disk_index[image_id])

    return sorted(set(pa_paths))


def _sample(paths: list, n: int, seed: int = 42) -> list:
    return random.Random(seed).sample(paths, min(n, len(paths)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Domain gap: NIH vs PadChest")
    parser.add_argument("--nih-dir", default="datasets/nih-chest-xrays")
    parser.add_argument("--padchest-dir", default="datasets/padchest")
    parser.add_argument("--n-samples", type=int, default=6,
                        help="Images to show per dataset in the sample grid")
    parser.add_argument("--n-hist", type=int, default=500,
                        help="Images to sample for histogram / statistics")
    parser.add_argument("--output", default="domain_gap.png")
    args = parser.parse_args()

    print("Collecting image paths...")
    nih_paths = collect_image_paths("nih", args.nih_dir)
    padchest_paths = _collect_padchest_pa(Path(args.padchest_dir))
    print(f"  NIH:      {len(nih_paths):,} images")
    print(f"  PadChest: {len(padchest_paths):,} PA/AP images")

    sample_nih = _sample(nih_paths, args.n_samples)
    sample_pc  = _sample(padchest_paths, args.n_samples)
    hist_nih   = _sample(nih_paths, args.n_hist)
    hist_pc    = _sample(padchest_paths, args.n_hist)

    print(f"Loading {args.n_hist} images per dataset for statistics...")
    nih_pixels = np.concatenate([_load_gray(p).ravel() for p in hist_nih])
    pc_pixels  = np.concatenate([_load_gray(p).ravel() for p in hist_pc])

    def stats(arr: np.ndarray, name: str) -> None:
        print(f"\n{name}:")
        print(f"  mean={arr.mean():.3f}  std={arr.std():.3f}  "
              f"p5={np.percentile(arr,5):.3f}  p95={np.percentile(arr,95):.3f}")

    stats(nih_pixels, "NIH pixel stats")
    stats(pc_pixels,  "PadChest pixel stats")

    mean_shift = abs(nih_pixels.mean() - pc_pixels.mean())
    std_ratio  = nih_pixels.std() / (pc_pixels.std() + 1e-8)
    print(f"\nDomain gap indicators:")
    print(f"  Mean shift:  {mean_shift:.3f}  (0=identical, >0.1=notable)")
    print(f"  Std ratio:   {std_ratio:.3f}  (1.0=identical)")

    n = args.n_samples
    fig = plt.figure(figsize=(14, 10))

    axes_nih = [fig.add_subplot(2, n * 2, i + 1)     for i in range(n)]
    axes_pc  = [fig.add_subplot(2, n * 2, n + i + 1) for i in range(n)]

    for ax, path in zip(axes_nih, sample_nih):
        ax.imshow(_load_gray(path), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    axes_nih[n // 2].set_title("NIH", fontsize=12, fontweight="bold", pad=8)

    for ax, path in zip(axes_pc, sample_pc):
        ax.imshow(_load_gray(path), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    axes_pc[n // 2].set_title("PadChest", fontsize=12, fontweight="bold", pad=8)

    ax_hist = fig.add_subplot(2, 1, 2)
    ax_hist.hist(nih_pixels, bins=100, alpha=0.6, color="steelblue",
                 density=True, label=f"NIH  (μ={nih_pixels.mean():.3f}, σ={nih_pixels.std():.3f})")
    ax_hist.hist(pc_pixels,  bins=100, alpha=0.6, color="tomato",
                 density=True, label=f"PadChest (μ={pc_pixels.mean():.3f}, σ={pc_pixels.std():.3f})")
    ax_hist.set_xlabel("Pixel intensity (normalised to [0,1])")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Pixel intensity distribution — NIH vs PadChest")
    ax_hist.legend()

    plt.tight_layout()
    out = Path(args.output)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out}")
    plt.show()
