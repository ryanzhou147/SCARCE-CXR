import argparse
import io
import os
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

TARGET_SIZE = 256

NIH_SLUG = "nih-chest-xrays/data"
NIH_DEST = "datasets/nih-chest-xrays"


def _resize_and_save(data: bytes, out_path: Path, size: int) -> None:
    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("L")
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    lo, hi = arr.min(), arr.max()
    # PadChest ships as 16-bit and may not use the full range
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype("uint8"), mode="L").save(out_path, format="PNG")


def _extract_and_resize(zip_path: Path, dest: Path, size: int) -> int:
    IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        for i, member in enumerate(members, 1):
            if member.is_dir():
                continue
            name = member.filename
            out_path = dest / name
            data = zf.read(member)
            if Path(name).suffix.lower() in IMAGE_EXTS:
                try:
                    _resize_and_save(data, out_path.with_suffix(".png"), size)
                    n += 1
                except Exception as e:
                    print(f"  WARN {name}: {e}")
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(data)
            if i % 5000 == 0:
                print(f"  {i}/{len(members)} ({n} images)...")
    zip_path.unlink()
    return n


def download(dest: Path, resize: bool = True) -> None:
    load_dotenv()
    dest.mkdir(parents=True, exist_ok=True)

    if len(list(dest.glob("**/*.png"))) > 100_000:
        print(f"NIH already downloaded ({dest})")
        return

    if not os.environ.get("KAGGLE_API_TOKEN"):
        print("ERROR: KAGGLE_API_TOKEN not set: add it to .env")
        sys.exit(1)

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    if resize:
        # unzip=False so full-res PNGs are never written to disk
        print("Downloading NIH (~45GB)...")
        api.dataset_download_files(dataset=NIH_SLUG, path=str(dest), unzip=False, quiet=False)
        total = 0
        for zp in sorted(dest.glob("*.zip")):
            print(f"Extracting + resizing {zp.name}...")
            total += _extract_and_resize(zp, dest, TARGET_SIZE)
        print(f"{total} images at {TARGET_SIZE}x{TARGET_SIZE} written to {dest}/")
    else:
        print("Downloading NIH (~45GB)...")
        api.dataset_download_files(dataset=NIH_SLUG, path=str(dest), unzip=True, quiet=False)
        print(f"{len(list(dest.glob('**/*.png')))} images in {dest}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=str, default=NIH_DEST)
    parser.add_argument("--no-resize", action="store_true")
    args = parser.parse_args()

    download(Path(args.dest), resize=not args.no_resize)
