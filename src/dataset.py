"""Data loading and augmentation transforms for the waste classifier."""
import platform
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T

from .config import IMG_SIZE, MEAN, STD


def train_transforms(size: int = IMG_SIZE):
    """Augmentations for training: random crops, flips, rotation, color jitter."""
    return T.Compose(
        [
            T.RandomResizedCrop(size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ]
    )


def eval_transforms(size: int = IMG_SIZE):
    """Fixed transforms for validation/test/inference."""
    return T.Compose(
        [
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ]
    )


def default_num_workers() -> int:
    """DataLoader workers. Windows spawn overhead -> use 0 by default."""
    return 0 if platform.system() == "Windows" else 2


def make_loaders(
    data_root,
    batch_size: int = 32,
    num_workers: int | None = None,
):
    """Build train/val/test ImageFolder loaders from data_root/{train,val,test}.

    Returns {"train": (loader, classes), "val": (loader, classes), ...}.
    """
    root = Path(data_root)
    if num_workers is None:
        num_workers = default_num_workers()

    splits = {}
    for name, split_dir in (
        ("train", root / "train"),
        ("val", root / "val"),
        ("test", root / "test"),
    ):
        if not split_dir.is_dir():
            continue
        transform = train_transforms() if name == "train" else eval_transforms()
        dataset = datasets.ImageFolder(str(split_dir), transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
        )
        splits[name] = (loader, dataset.classes)

    if "train" not in splits:
        raise FileNotFoundError(
            f"No 'train' folder found under {root}. "
            "Run 'py scripts\\prepare_data.py' first."
        )
    return splits
