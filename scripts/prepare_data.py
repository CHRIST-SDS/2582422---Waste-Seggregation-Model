"""Split raw TrashNet class folders into a train/val/test ImageFolder structure.

Creates data/{train,val,test}/<class>/... (70/15/15 split, seeded).

Usage:
    py scripts\\prepare_data.py
"""
import argparse
import random
import shutil
from pathlib import Path

from src.config import RAW_DATA_DIR, TEST_DIR, TRAIN_DIR, VAL_DIR


def split_class(src_dir: Path, seed: int):
    files = sorted(p for p in src_dir.iterdir() if p.is_file())
    if not files:
        return 0
    rng = random.Random(seed)
    rng.shuffle(files)
    n = len(files)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    groups = [
        (TRAIN_DIR, files[:n_train]),
        (VAL_DIR, files[n_train : n_train + n_val]),
        (TEST_DIR, files[n_train + n_val :]),
    ]
    for dest, subset in groups:
        class_dest = dest / src_dir.name
        class_dest.mkdir(parents=True, exist_ok=True)
        for f in subset:
            shutil.copy2(f, class_dest / f.name)
    return n


def main():
    parser = argparse.ArgumentParser(description="Split raw data into train/val/test")
    parser.add_argument("--source", default=str(RAW_DATA_DIR))
    parser.add_argument("--force", action="store_true", help="recreate split dirs from scratch")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        raise SystemExit(
            f"Source dir not found: {source}\nRun 'py scripts\\download_trashnet.py' first."
        )

    if args.force:
        for d in (TRAIN_DIR, VAL_DIR, TEST_DIR):
            if d.exists():
                shutil.rmtree(d)

    print("Splitting dataset (70/15/15)...")
    classes = sorted(d.name for d in source.iterdir() if d.is_dir())
    if not classes:
        raise SystemExit(f"No class folders found under {source}.")
    for cls in classes:
        cls_dir = source / cls
        if not cls_dir.is_dir():
            print(f"  WARNING: missing class folder {cls_dir}")
            continue
        n = split_class(cls_dir, seed=42)
        print(f"  {cls}: {n} images -> train/val/test")

    print("Done.")
    for name, d in (("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)):
        total = sum(len(list(p.iterdir())) for p in d.iterdir() if p.is_dir())
        print(f"  {name}: {total} images in {d}")


if __name__ == "__main__":
    main()
