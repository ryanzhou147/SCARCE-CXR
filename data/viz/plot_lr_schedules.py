"""Generate learning rate schedule SVGs for the four SSL methods.

uv run python -m data.viz.plot_lr_schedules
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_STATIC = Path(__file__).parent.parent.parent.parent / "personal-website/portfolio/static/clear-cxr"

_RC = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#d0cec8",
    "axes.labelcolor":   "#555555",
    "axes.titlecolor":   "#333333",
    "axes.titlesize":    15,
    "axes.labelsize":    13,
    "axes.grid":         True,
    "grid.color":        "#e8e6e0",
    "grid.linewidth":    0.7,
    "grid.linestyle":    "--",
    "xtick.color":       "#888888",
    "ytick.color":       "#888888",
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "text.color":        "#555555",
    "savefig.facecolor": "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "lines.linewidth":   1.8,
    "font.family":       "sans-serif",
    "legend.fontsize":   11,
}

_BLUE = "#6b8cba"


def _lr_schedule(base_lr: float, warmup: int, total: int) -> tuple[list[int], list[float]]:
    epochs = list(range(total))
    lrs = []
    for e in epochs:
        if e < warmup:
            factor = (e + 1) / warmup
        else:
            progress = (e - warmup) / max(1, total - warmup)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        lrs.append(base_lr * factor)
    return epochs, lrs


def _save(fig: plt.Figure, name: str) -> None:
    _STATIC.mkdir(parents=True, exist_ok=True)
    out = _STATIC / name
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_moco() -> None:
    epochs, lrs = _lr_schedule(4.5e-2, warmup=10, total=800)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(epochs, lrs, color=_BLUE)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_title("MoCo v2: learning rate schedule")
        fig.tight_layout()
    _save(fig, "moco_lr.svg")


def plot_barlow() -> None:
    epochs, lrs = _lr_schedule(2.4e-3, warmup=10, total=200)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(epochs, lrs, color=_BLUE)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_title("BarlowTwins: learning rate schedule")
        fig.tight_layout()
    _save(fig, "barlow_lr.svg")


def plot_spark() -> None:
    epochs, lrs = _lr_schedule(3.0e-4, warmup=20, total=200)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(epochs, lrs, color=_BLUE)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_title("SparK: learning rate schedule")
        fig.tight_layout()
    _save(fig, "spark_lr.svg")


def plot_dino() -> None:
    stop_epoch = 15
    epochs, lrs = _lr_schedule(5.0e-4, warmup=10, total=100)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5, 3))

        # Projected full run (dashed)
        ax.plot(epochs, lrs, color=_BLUE, linestyle="--", linewidth=1.4,
                label="projected (full 100 ep)")

        # Actual run (solid up to stop_epoch)
        ax.plot(epochs[:stop_epoch + 1], lrs[:stop_epoch + 1], color=_BLUE, linewidth=1.8,
                label="actual (wandb)")

        # Vertical marker at stop epoch
        ax.axvline(x=stop_epoch, color="#cc4444", linestyle="--", linewidth=1.2)
        ax.text(stop_epoch + 1.5, max(lrs) * 0.72,
                f"loss plateaued\n(ep {stop_epoch})",
                color="#cc4444", fontsize=11)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_title("DINO: learning rate schedule")
        ax.legend(loc="lower right")
        fig.tight_layout()
    _save(fig, "dino_lr.svg")


def main() -> None:
    plot_moco()
    plot_barlow()
    plot_spark()
    plot_dino()


if __name__ == "__main__":
    main()
