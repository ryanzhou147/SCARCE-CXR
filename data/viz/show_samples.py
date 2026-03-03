"""
Display grid of sample chest X-rays from the dataset.
uv run python -m data.viz.show_samples --data-dir datasets/nih-chest-xrays
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

DEFAULT_DATA_DIR = "datasets/nih-chest-xrays"
GRID = 5
TOTAL = GRID * GRID


def reservoir_sample(iterator, k: int, rng: random.Random) -> list:
    """Pick k items from an iterator without loading it all into memory."""
    result = []
    for i, item in enumerate(iterator):
        if i < k:
            result.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                result[j] = item
    return result


def main():
    parser = argparse.ArgumentParser(description="Show sample chest X-ray grid")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    rng = random.Random(42)
    sample = reservoir_sample(data_dir.glob("**/*.png"), TOTAL, rng)
    print(f"Showing {len(sample)} images")

    fig, axes = plt.subplots(GRID, GRID, figsize=(10, 10))
    fig.suptitle(f"Sample Chest X-rays — {len(sample)} images", fontsize=20)

    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(sample):
            img = Image.open(sample[i])
            img.thumbnail((128, 128))
            ax.imshow(img, cmap="gray")

    plt.tight_layout()
    out_path = data_dir / "sample_grid.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    print(f"Saved grid to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
