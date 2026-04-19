"""Regenerate all probe and finetune per-disease PNG plots from saved state files.

Usage:
    uv run python -m data.viz.replot
"""

import json
import re
from pathlib import Path

from finetune._plots import plot_mean_auc, plot_per_disease

STATIC_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "personal-website/portfolio/static/clear-cxr"
)
PROJECT_ROOT = Path(__file__).parent.parent.parent

FEW_SHOT_NS = [1, 5, 10, 20, 50]


def _parse_probe_md(md_path: Path) -> list[tuple[str, int, dict]]:
    """Parse probe results markdown into [(mname, epoch, all_results), ...]."""
    text = md_path.read_text()
    # split on top-level ## sections
    sections = re.split(r"\n## ", "\n" + text)

    out = []
    for section in sections:
        header_line = section.split("\n")[0].strip()
        if not header_line or header_line.startswith("#"):
            continue

        # e.g. "moco-v2 (ResNet50, ep774)"
        if "moco" in header_line.lower():
            mname = "moco"
        elif "barlow" in header_line.lower():
            mname = "barlow"
        elif "spark" in header_line.lower():
            mname = "spark"
        else:
            continue

        epoch_match = re.search(r"ep(\d+)", header_line)
        if not epoch_match:
            continue
        epoch = int(epoch_match.group(1))

        all_results: dict[str, dict] = {}
        init_name: str | None = None

        for line in section.split("\n"):
            if line.startswith("### SSL"):
                init_name = "SSL pretrained"
                all_results[init_name] = {}
            elif line.startswith("### ImageNet"):
                init_name = "ImageNet"
                all_results[init_name] = {}
            elif line.startswith("### Random"):
                init_name = "Random init"
                all_results[init_name] = {}
            elif init_name and line.startswith("|"):
                cols = [c.strip() for c in line.strip("|").split("|")]
                if len(cols) < 6 or cols[0] in ("Disease", "") or cols[0].startswith("-"):
                    continue
                disease = cols[0]
                all_results[init_name][disease] = {}
                for j, n in enumerate(FEW_SHOT_NS):
                    m = re.match(r"([\d.]+)±([\d.]+)", cols[j + 1].strip())
                    if m:
                        all_results[init_name][disease][n] = {
                            "auc": (float(m.group(1)), float(m.group(2)))
                        }

        if all_results:
            out.append((mname, epoch, all_results))

    return out


def replot_probes() -> None:
    md_path = PROJECT_ROOT / "results/probe_padchest_2026-03-24.md"
    if not md_path.exists():
        print(f"Probe results not found: {md_path}")
        return

    for mname, epoch, all_results in _parse_probe_md(md_path):
        classes = list(list(all_results.values())[0].keys())
        out = STATIC_DIR / f"{mname}_ep{epoch}_probe_binary.png"
        out2 = STATIC_DIR / f"{mname}_ep{epoch}_probe_per_disease.png"
        print(f"  probe {mname} ep{epoch}...")
        plot_mean_auc(all_results, classes, FEW_SHOT_NS, mname, epoch, out, title_prefix="Probe")
        plot_per_disease(all_results, classes, FEW_SHOT_NS, mname, epoch, out2, title_prefix="Probe")


def replot_finetune() -> None:
    for state_path in sorted(PROJECT_ROOT.glob("*_finetune_state.json")):
        m = re.match(r"(\w+)_ep(\d+)_finetune_state", state_path.stem)
        if not m:
            print(f"  skipping {state_path.name}: can't parse mname/epoch")
            continue
        mname, epoch = m.group(1), int(m.group(2))

        raw = json.loads(state_path.read_text())["results"]
        all_results: dict[str, dict] = {
            init: {
                disease: {int(n): v for n, v in ns_data.items()}
                for disease, ns_data in diseases.items()
            }
            for init, diseases in raw.items()
        }

        classes = list(list(all_results.values())[0].keys())
        ns = sorted(
            int(n)
            for n in list(list(raw.values())[0].values())[0]
            if int(n) != -1
        )

        out = STATIC_DIR / f"{mname}_ep{epoch}_finetune_binary.png"
        out2 = STATIC_DIR / f"{mname}_ep{epoch}_finetune_per_disease.png"
        print(f"  finetune {mname} ep{epoch}...")
        plot_mean_auc(all_results, classes, ns, mname, epoch, out, title_prefix="Finetune")
        plot_per_disease(all_results, classes, ns, mname, epoch, out2, title_prefix="Finetune")


def replot_combined() -> None:
    """Generate one combined probe+finetune per-disease plot per method."""
    probe_md = PROJECT_ROOT / "results/probe_padchest_2026-03-24.md"
    if not probe_md.exists():
        print("Probe results not found, skipping combined")
        return

    probe_by_mname = {mname: (epoch, res) for mname, epoch, res in _parse_probe_md(probe_md)}

    for state_path in sorted(PROJECT_ROOT.glob("*_finetune_state.json")):
        m = re.match(r"(\w+)_ep(\d+)_finetune_state", state_path.stem)
        if not m:
            continue
        mname, epoch = m.group(1), int(m.group(2))

        if mname not in probe_by_mname:
            print(f"  No probe data for {mname}, skipping")
            continue

        _epoch_probe, probe_results = probe_by_mname[mname]

        raw = json.loads(state_path.read_text())["results"]
        finetune_results: dict[str, dict] = {
            init: {
                disease: {int(n): v for n, v in ns_data.items()}
                for disease, ns_data in diseases.items()
            }
            for init, diseases in raw.items()
        }

        classes = list(list(probe_results.values())[0].keys())
        ns = sorted(int(n) for n in list(list(raw.values())[0].values())[0] if int(n) != -1)

        out = STATIC_DIR / f"{mname}_ep{epoch}_combined_per_disease.png"
        print(f"  combined {mname} ep{epoch}...")
        plot_per_disease(
            probe_results, classes, ns, mname, epoch, out,
            title_prefix="Probe", secondary_results=finetune_results,
        )


if __name__ == "__main__":
    print("Regenerating probe plots...")
    replot_probes()
    print("Regenerating finetune plots...")
    replot_finetune()
    print("Regenerating combined probe+finetune plots...")
    replot_combined()
    print("Done.")
