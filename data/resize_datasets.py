import argparse
import multiprocessing as mp
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

_DEFAULT_SIZE = 256


def _resize_one(args: tuple[Path, int]) -> tuple[Path, str]:
    path, target = args
    try:
        img = Image.open(path)
        if img.size == (target, target) and img.mode == "L":
            return path, "skip"

        arr = np.array(img).astype(np.float32)
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255
        arr = arr.astype(np.uint8)

        out = Image.fromarray(arr).convert("L").resize((target, target), Image.BILINEAR)

        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        out.save(tmp_path, format="PNG")
        os.replace(tmp_path, path)
        return path, "ok"

    except Exception as e:
        return path, f"error: {e}"


def _resize_dir(root: Path, target: int, dry_run: bool, workers: int) -> None:
    paths = sorted(root.rglob("*.png"))
    print(f"  {root}: {len(paths):,} PNGs found")

    if dry_run:
        if paths:
            img = Image.open(paths[0])
            print(f"  Sample: {paths[0].name}  size={img.size}  mode={img.mode}")
        print("  --dry-run: no files written")
        return

    ok = skip = errors = 0
    with mp.Pool(workers) as pool:
        work = [(p, target) for p in paths]
        for i, (path, status) in enumerate(pool.imap_unordered(_resize_one, work, chunksize=64)):
            if status == "ok":       ok += 1
            elif status == "skip":   skip += 1
            else:
                errors += 1
                print(f"  WARN {path.name}: {status}")
            if (i + 1) % 5000 == 0 or (i + 1) == len(paths):
                print(f"  {i+1:,}/{len(paths):,}  ok={ok}  skip={skip}  err={errors}")

    print(f"  Done: {ok} resized, {skip} already {target}px, {errors} errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="PadChest directory to resize")
    parser.add_argument("--size", type=int, default=_DEFAULT_SIZE)
    parser.add_argument("--workers", type=int, default=min(8, mp.cpu_count()))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        print(f"Resizing images IN-PLACE to {args.size}×{args.size} 8-bit grayscale.")
        print(f"Workers: {args.workers}\n")

    _resize_dir(Path(args.dir), args.size, args.dry_run, args.workers)

