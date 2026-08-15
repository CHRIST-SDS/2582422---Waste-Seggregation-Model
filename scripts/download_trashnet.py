"""Download the TrashNet resized dataset (6 classes, ~43 MB) from the official GitHub repo.

Usage:
    py scripts\\download_trashnet.py
"""
import argparse
import io
import zipfile
from pathlib import Path

import requests

URL = "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip"
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def main():
    parser = argparse.ArgumentParser(description="Download the TrashNet resized dataset")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "data"),
        help="output directory (default: project data/)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "dataset-resized"

    if dest.is_dir() and all((dest / c).is_dir() for c in CLASSES):
        print(f"Dataset already present at {dest}. Skipping download.")
        return

    print(f"Downloading TrashNet resized dataset from:\n  {URL}")
    response = requests.get(URL, stream=True, timeout=300)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        for info in zf.infolist():
            parts = Path(info.filename).parts
            if not parts or parts[0] in {"__MACOSX", ""}:
                continue
            if info.is_dir():
                continue
            target = out_dir / Path(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))

    # The zip extracts a top-level folder such as "dataset-resized".
    candidates = [d for d in out_dir.iterdir() if d.is_dir() and d.name != "dataset-resized"]
    for cand in candidates:
        if all((cand / c).is_dir() for c in CLASSES):
            if dest.exists():
                import shutil

                shutil.rmtree(dest)
            cand.rename(dest)
            break

    missing = [c for c in CLASSES if not (dest / c).is_dir()]
    if missing:
        raise RuntimeError(f"Download/extract incomplete; missing class folders: {missing}")
    print(f"Dataset ready at {dest}")
    for cls in CLASSES:
        print(f"  {cls}: {sum(1 for _ in (dest / cls).iterdir())} images")


if __name__ == "__main__":
    main()
