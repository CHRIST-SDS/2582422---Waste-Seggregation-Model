"""Training entry point.

Usage:
    py -m src.train --arch resnet18 --epochs 25
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from .config import (
    DATA_DIR,
    DEFAULT_ARCH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    MODELS_DIR,
)
from .dataset import make_loaders
from .model import ARCHS, build_model


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return running_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Train a waste-image classifier")
    parser.add_argument("--data", default=str(DATA_DIR), help="dir containing train/val/test subfolders")
    parser.add_argument("--arch", default=DEFAULT_ARCH, choices=sorted(ARCHS))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(MODELS_DIR))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"Device: {device}")

    splits = make_loaders(args.data, batch_size=args.batch_size)
    train_loader, class_names = splits["train"]
    val_loader = splits.get("val") and splits["val"][0]
    test_loader = splits.get("test") and splits["test"][0]
    print(f"Classes: {class_names}")

    model = build_model(args.arch, num_classes=len(class_names), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        else:
            val_loss, val_acc = float("nan"), 0.0
        scheduler.step()
        print(
            f"Epoch {epoch}/{args.epochs}  train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | {time.time() - t0:.0f}s"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = out_dir / "best_model.pth"
            torch.save(
                {
                    "arch": args.arch,
                    "class_names": class_names,
                    "state_dict": model.state_dict(),
                    "val_acc": best_acc,
                },
                ckpt_path,
            )
            print(f"  -> saved best model to {ckpt_path} (val acc {best_acc:.4f})")

    if test_loader is not None:
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Final test loss {test_loss:.4f} | test accuracy {test_acc:.4f}")

    print(f"Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
