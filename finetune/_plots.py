import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from finetune._data import COLORS, PLOT_RC

plt.rcParams.update(PLOT_RC)


def plot_mean_auc(
    all_results: dict,
    classes: list[str],
    ns: list[int],
    mname: str,
    epoch: int,
    out_path: Path,
    title_prefix: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for init_name in all_results:
        means = [np.mean([all_results[init_name][d][n]["auc"][0] for d in classes]) for n in ns]
        stds  = [np.mean([all_results[init_name][d][n]["auc"][1] for d in classes]) for n in ns]
        ax.plot(ns, means, marker="o", label=init_name, color=COLORS[init_name])
        ax.fill_between(
            ns,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.15, color=COLORS[init_name],
        )
    prefix = f"{title_prefix} — " if title_prefix else ""
    ax.set_xlabel("Shots per disease")
    ax.set_ylabel("Mean AUC (binary, across diseases)")
    ax.set_title(f"{prefix}{mname} ep{epoch}\n{len(classes)} diseases")
    ax.legend()
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.close()


def plot_per_disease(
    all_results: dict,
    classes: list[str],
    ns: list[int],
    mname: str,
    epoch: int,
    out_path: Path,
    title_prefix: str = "",
) -> None:
    n_cols = 5
    n_rows = math.ceil(len(classes) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).flatten()

    for i, disease in enumerate(classes):
        ax = axes[i]
        for init_name in all_results:
            means = [all_results[init_name][disease][n]["auc"][0] for n in ns]
            stds  = [all_results[init_name][disease][n]["auc"][1] for n in ns]
            ax.plot(ns, means, marker="o", label=init_name, color=COLORS[init_name], linewidth=1.5)
            ax.fill_between(
                ns,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.12, color=COLORS[init_name],
            )
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(disease, fontsize=9)
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels(ns, fontsize=7)
        ax.set_ylim(0.3, 1.05)
        ax.grid(True, alpha=0.2)
        if i % n_cols == 0:
            ax.set_ylabel("AUC", fontsize=8)

    for j in range(len(classes), len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=9)
    prefix = f"{title_prefix} — " if title_prefix else ""
    fig.suptitle(f"{prefix}per-disease AUC — {mname} ep{epoch}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved per-disease plot to {out_path}")
    plt.close()
