"""Download organic-waste images from the AlphaTrash dataset (GitHub).

Pulls only the `organic` class (food scraps, peels) into
data/dataset-resized/organic/ so it can be trained alongside TrashNet.
Sourced from https://github.com/Patipol-BKK/alphatrash-dataset (MIT).

Usage:
    py scripts\\download_organic.py [--max-images 400] [--workers 8]
"""
import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REPO = "Patipol-BKK/alphatrash-dataset"
BRANCH = "main"
TREE_URL = f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/"


def list_organic_paths() -> list[str]:
    with urllib.request.urlopen(TREE_URL, timeout=120) as r:
        tree = json.load(r)["tree"]
    paths = [
        t["path"]
        for t in tree
        if "organic" in t["path"].lower() and t["path"].lower().endswith((".jpeg", ".jpg", ".png"))
    ]
    return paths


def fetch_one(session: requests.Session, dest: Path, remote: str, i: int, total: int):
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    try:
        with session.get(RAW_URL + remote, timeout=60) as resp:
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"fail:{exc}"


def main():
    parser = argparse.ArgumentParser(description="Download organic waste images (AlphaTrash)")
    parser.add_argument(
        "--max-images", type=int, default=0,
        help="cap number of images to download (0 = all)",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "data" / "dataset-resized" / "organic"),
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_idx = set()
    for p in out_dir.iterdir():
        if p.is_file():
            try:
                existing_idx.add(int(p.stem.rsplit("_", 1)[-1]))
            except ValueError:
                pass
    print(f"Found {len(existing_idx)} existing images in {out_dir}; filling the rest.")

    print("Listing organic images on GitHub...")
    paths = list_organic_paths()
    limit = args.max_images if args.max_images > 0 else len(paths)
    targets = [i for i in range(limit) if i not in existing_idx]
    print(f"Downloading {len(targets)} of {limit} organic images (may take a few minutes)...")

    session = requests.Session()
    stats = {"ok": 0, "skip": 0, "fail": 0}

    def job(idx):
        remote = paths[idx]
        dest = out_dir / f"organic_{idx:04d}.jpeg"
        return idx, dest, fetch_one(session, dest, remote, idx, len(paths))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, i) for i in targets]
        for future in as_completed(futures):
            idx, dest, result = future.result()
            if result == "ok":
                stats["ok"] += 1
            elif result == "skip":
                stats["skip"] += 1
            else:
                stats["fail"] += 1
                print(f"  {idx}: {result}")
            if (stats["ok"] + stats["skip"] + stats["fail"]) % 100 == 0:
                print(f"  ... {stats}")

    print(f"Done: {stats}")
    if stats["fail"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
