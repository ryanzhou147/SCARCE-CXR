"""Count PadChest disease labels with train/val split breakdown.

Applies the same patient-level split and exclusion filters as finetune.py
so the counts shown here exactly match what finetune will use.

Why a label can fail even when train+val total > 23:
  The split is patient-level (not image-level). 20% of *patients* go to val.
  A rare disease affecting only a few patients can end up with all of them in
  train by chance, leaving val with 0-2 examples — below the MIN_VAL threshold.
  Total count doesn't matter; both splits must independently meet their minimums.
  To fix: download more zips covering that disease, or lower MIN_TEST_PER_CLASS.

Usage:
  uv run python -m finetune.count_labels
  uv run python -m finetune.count_labels --data-dir datasets/padchest --val-frac 0.2
"""

import argparse
import ast
import csv
from pathlib import Path

from finetune._data import EXCLUDE_LABELS, MIN_TEST_PER_CLASS, MIN_TRAIN_PER_CLASS


def count_labels(data_dir: Path, val_frac: float, seed: int) -> None:
    # Build disk index — scans all subfolders (21/, 23/, 27/, etc.)
    all_pngs = list(data_dir.rglob("*.png"))
    disk_index = {p.name: p for p in all_pngs}
    subdirs = sorted({p.parent.name for p in all_pngs if p.parent != data_dir})
    print(f"Images on disk: {len(disk_index)}  (subfolders: {', '.join(subdirs) or 'root only'})")

    # Find CSV
    gz_files = sorted(data_dir.glob("*.csv.gz"))
    csv_files = sorted(data_dir.glob("*.csv"))
    if not gz_files and not csv_files:
        raise FileNotFoundError(f"No CSV found in {data_dir}")

    def open_csv():
        if gz_files:
            import gzip
            return gzip.open(gz_files[0], "rt")
        return open(csv_files[0])

    # Collect single-label PA/AP images grouped by patient
    by_patient: dict[str, list[tuple[str, str]]] = {}
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
            if not label or label in EXCLUDE_LABELS:
                continue
            patient_id = row.get("PatientID", "").strip()
            by_patient.setdefault(patient_id, []).append((image_id, label))

    # Patient-level train/val split — hash-based so the assignment is stable
    # regardless of how many zips are on disk.
    import hashlib
    val_patients = {
        pid for pid in by_patient
        if int(hashlib.md5(pid.encode()).hexdigest(), 16) % round(1 / val_frac) == 0
    }
    train_counts: dict[str, int] = {}
    val_counts: dict[str, int] = {}
    for patient_id, samples in by_patient.items():
        target = val_counts if patient_id in val_patients else train_counts
        for _, label in samples:
            target[label] = target.get(label, 0) + 1

    # Print all labels sorted by total count descending
    all_labels = sorted(
        set(train_counts) | set(val_counts),
        key=lambda l: -(train_counts.get(l, 0) + val_counts.get(l, 0)),
    )

    n_val = len(val_patients)
    print(f"\nPatients: {len(by_patient)} total  |  train={len(by_patient)-n_val}  val={n_val}")
    print(f"\n{'label':<45s}  {'train':>6s}  {'val':>5s}  {'total':>6s}  passes")
    print("-" * 75)

    passing = []
    for label in all_labels:
        tr = train_counts.get(label, 0)
        vl = val_counts.get(label, 0)
        ok = tr >= MIN_TRAIN_PER_CLASS and vl >= MIN_TEST_PER_CLASS
        flag = "  YES" if ok else f"  -- (need train>={MIN_TRAIN_PER_CLASS}, val>={MIN_TEST_PER_CLASS})"
        print(f"  {label:<43s}  {tr:>6d}  {vl:>5d}  {tr+vl:>6d}{flag}")
        if ok:
            passing.append(label)

    print(f"\n{len(passing)} classes pass threshold (MIN_TRAIN={MIN_TRAIN_PER_CLASS}, MIN_VAL={MIN_TEST_PER_CLASS}):")
    for label in passing:
        print(f"  {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Count PadChest disease labels")
    parser.add_argument("--data-dir", type=str, default="datasets/padchest")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    count_labels(Path(args.data_dir), args.val_frac, args.seed)


if __name__ == "__main__":
    main()
