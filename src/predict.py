"""Inference utilities: load a checkpoint and classify an image or directory.

Usage:
    py -m src.predict --checkpoint models/best_model.pth --image path/to/image.jpg
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .config import CLASS_NAMES, IMG_SIZE, LOW_CONFIDENCE_THRESHOLD
from .dataset import eval_transforms
from .model import build_model

ImageSource = str | Path | Image.Image


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path, device=None):
    """Load a checkpoint saved by src/train.py. Returns (model, class_names, device)."""
    device = device or default_device()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not (isinstance(ckpt, dict) and "state_dict" in ckpt):
        raise ValueError("Unsupported checkpoint format. Train with 'py -m src.train' first.")

    arch = ckpt.get("arch", "resnet18")
    class_names = ckpt.get("class_names", CLASS_NAMES)
    model = build_model(arch=arch, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    return model, class_names, device


def predict_image(model, class_names, source: ImageSource, top_k: int = 3, device=None):
    """Classify a single image (path or PIL image). Returns a structured dict."""
    device = device or default_device()
    if isinstance(source, (str, Path)):
        image = Image.open(source).convert("RGB")
    else:
        image = source.convert("RGB")

    tensor = eval_transforms(IMG_SIZE)(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu()

    values, indices = probs.topk(top_k)
    probas = {class_names[i]: float(probs[i]) for i in indices.tolist()}
    label = class_names[indices[0].item()]
    confidence = float(values[0])
    return {
        "label": label,
        "confidence": confidence,
        "probas": probas,
        "low_confidence": confidence < LOW_CONFIDENCE_THRESHOLD,
    }


def main():
    parser = argparse.ArgumentParser(description="Classify waste images with a trained model")
    parser.add_argument("--checkpoint", required=True, help="path to best_model.pth")
    parser.add_argument("--image", help="single image path")
    parser.add_argument("--dir", help="directory of images to classify")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.error("provide --image or --dir")

    model, class_names, device = load_model(args.checkpoint)

    def show(src, is_path=True):
        pred = predict_image(model, class_names, src, top_k=args.top_k, device=device)
        if args.json:
            return pred
        lines = [
            f"Prediction: {pred['label']}  (confidence {pred['confidence'] * 100:.1f}%)",
            "Top classes: " + ", ".join(
                f"{k} {v * 100:.1f}%" for k, v in pred["probas"].items()
            ),
            "Ambiguous (low confidence)" if pred["low_confidence"] else "Confident prediction",
        ]
        return "\n".join(lines)

    if args.image:
        print(show(args.image))
    else:
        results = {}
        for img_path in sorted(Path(args.dir).iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            results[img_path.name] = show(img_path)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for name, text in results.items():
                print(f"{name}:\n{text}\n")


if __name__ == "__main__":
    main()
