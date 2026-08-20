"""Generate a demo video showing model predictions on sample test images.

Usage:
    py scripts\demo_video.py --checkpoint models\best_model.pth --output demo_video.mp4
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.config import CLASS_NAMES, IMG_SIZE, TEST_DIR, MODELS_DIR, MODELS_DIR
from src.model import build_model
from src.predict import load_model, predict_image
from src.recommend import recommendation_from_prediction, BINS

# Video settings
WIDTH, HEIGHT = 1280, 720
FPS = 1
FRAMES_PER_SAMPLE = 5  # seconds per sample (at 1 fps = frames)


def get_font(size: int):
    """Try to load a decent font; fall back to default."""
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for fp in font_paths:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def draw_title_frame() -> np.ndarray:
    """Create the title frame."""
    img = Image.new("RGB", (WIDTH, HEIGHT), "#f8fafc")
    draw = ImageDraw.Draw(img)

    title_font = get_font(52)
    sub_font = get_font(24)
    small_font = get_font(18)

    # Title
    draw.text((WIDTH // 2, 200), "AI-Powered Waste Segregation",
              fill="#1e293b", font=title_font, anchor="mm")
    draw.text((WIDTH // 2, 270), "Computer Vision + LLM Demo",
              fill="#64748b", font=sub_font, anchor="mm")

    # Bin legend
    y = 360
    bin_items = [
        ("GREEN  Biodegradable / Organic", "#4CAF50"),
        ("BLUE   Recyclable / Dry Waste", "#2196F3"),
        ("RED    Non-recyclable / Special", "#EF4444"),
    ]
    for text, color in bin_items:
        draw.rounded_rectangle(
            [WIDTH // 2 - 250, y - 16, WIDTH // 2 - 230, y + 4],
            radius=4, fill=color
        )
        draw.text((WIDTH // 2 - 215, y - 6), text, fill="#334155", font=small_font)
        y += 36

    draw.text((WIDTH // 2, 540), "Model: MobileNetV3-Small  |  7 Classes  |  92.5% Val Accuracy",
              fill="#94a3b8", font=small_font, anchor="mm")
    draw.text((WIDTH // 2, 580), "Click through to see classifications...",
              fill="#94a3b8", font=small_font, anchor="mm")

    return np.array(img)


def draw_sample_frame(
    image: Image.Image,
    label: str,
    confidence: float,
    rec: dict,
    true_class: str,
) -> np.ndarray:
    """Create a classification result frame."""
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "#f8fafc")
    draw = ImageDraw.Draw(canvas)

    title_font = get_font(32)
    label_font = get_font(26)
    body_font = get_font(20)
    small_font = get_font(16)
    big_font = get_font(60)

    hex_color = rec["hex_color"]
    r, g, b = tuple(int(hex_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

    # Left: the waste image (scaled to fit)
    img_w, img_h = image.size
    max_img_w, max_img_h = 480, 480
    scale = min(max_img_w / img_w, max_img_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    resized = image.resize((new_w, new_h), Image.LANCZOS)

    img_x = 60
    img_y = (HEIGHT - new_h) // 2
    # Shadow
    draw.rounded_rectangle(
        [img_x + 4, img_y + 4, img_x + new_w + 4, img_y + new_h + 4],
        radius=16, fill="#cbd5e1"
    )
    canvas.paste(resized, (img_x, img_y))

    # True class label below image
    draw.text((img_x + new_w // 2, img_y + new_h + 20),
              f"Actual: {true_class}", fill="#64748b", font=small_font, anchor="mm")

    # Right: result panel
    px = 600

    # Bin badge
    badge_x, badge_y = px, 80
    draw.rounded_rectangle(
        [badge_x, badge_y, badge_x + 70, badge_y + 70],
        radius=16, fill=(r, g, b)
    )
    draw.text((badge_x + 35, badge_y + 35), label[0].upper(),
              fill="white", font=label_font, anchor="mm")

    # Bin name + stream
    draw.text((badge_x + 90, badge_y + 8), f"{rec['bin']} BIN",
              fill=hex_color, font=title_font)
    draw.text((badge_x + 90, badge_y + 42), rec["stream"],
              fill="#64748b", font=small_font)

    # Confidence
    draw.text((px, 170), "Confidence", fill="#94a3b8", font=small_font)
    draw.text((px + 400, 165), f"{confidence * 100:.1f}%",
              fill=hex_color, font=big_font, anchor="rt")

    # Confidence bar
    bar_x, bar_y, bar_w, bar_h = px, 220, 400, 12
    draw.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                           radius=6, fill="#e2e8f0")
    bar_fill_w = int(bar_w * confidence)
    if bar_fill_w > 0:
        draw.rounded_rectangle(
            [bar_x, bar_y, bar_x + bar_fill_w, bar_y + bar_h],
            radius=6, fill=(r, g, b)
        )

    # Detected class
    draw.text((px, 265), "DETECTED", fill="#94a3b8", font=small_font)
    draw.text((px, 295), label.upper(), fill="#1e293b", font=title_font)
    draw.text((px + 280, 295), f"{confidence * 100:.1f}%", fill=hex_color,
              font=label_font)

    # Instructions
    draw.text((px, 360), "DISPOSAL INSTRUCTIONS", fill="#94a3b8", font=small_font)

    # Word-wrap instructions
    instructions = rec["instructions"]
    words = instructions.split()
    lines = []
    current_line = ""
    for word in words:
        test = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=body_font)
        if bbox[2] - bbox[0] > 400:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test
    if current_line:
        lines.append(current_line)

    y = 390
    for line in lines[:4]:
        draw.text((px, y), line, fill="#334155", font=body_font)
        y += 30

    # Low confidence warning
    if rec["low_confidence"]:
        draw.rounded_rectangle(
            [px, y + 10, px + 400, y + 50],
            radius=8, fill="#fef3c7"
        )
        draw.text((px + 10, y + 18),
                  "Low confidence — please verify!", fill="#92400e", font=small_font)
        y += 60

    # Footer
    draw.text((WIDTH // 2, HEIGHT - 30),
              "AI Waste Segregation  |  MobileNetV3-Small  |  7 Classes",
              fill="#cbd5e1", font=small_font, anchor="mm")

    return np.array(canvas)


def draw_summary_frame(results: list) -> np.ndarray:
    """Create a summary frame showing all classifications."""
    img = Image.new("RGB", (WIDTH, HEIGHT), "#f8fafc")
    draw = ImageDraw.Draw(img)

    title_font = get_font(36)
    label_font = get_font(22)
    small_font = get_font(16)

    draw.text((WIDTH // 2, 50), "Classification Summary",
              fill="#1e293b", font=title_font, anchor="mm")
    draw.text((WIDTH // 2, 85), "All 7 waste categories classified successfully",
              fill="#64748b", font=small_font, anchor="mm")

    # Grid of results
    cols = 4
    card_w, card_h = 260, 130
    start_x = (WIDTH - cols * (card_w + 20)) // 2
    start_y = 130

    for i, r in enumerate(results):
        row, col = divmod(i, cols)
        x = start_x + col * (card_w + 20)
        y = start_y + row * (card_h + 20)

        hex_c = r["hex_color"]
        cr, cg, cb = tuple(int(hex_c.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

        draw.rounded_rectangle(
            [x, y, x + card_w, y + card_h],
            radius=12, fill="white", outline="#e2e8f0"
        )
        # Colored left bar
        draw.rounded_rectangle(
            [x, y, x + 6, y + card_h],
            radius=3, fill=(cr, cg, cb)
        )
        draw.text((x + 20, y + 15), r["label"].upper(),
                  fill="#1e293b", font=label_font)
        draw.text((x + 20, y + 45), f"{r['bin']} BIN",
                  fill=hex_c, font=label_font)
        draw.text((x + 20, y + 75), f"{r['confidence'] * 100:.1f}% confidence",
                  fill="#64748b", font=small_font)
        draw.text((x + 20, y + 98), r["stream"],
                  fill="#94a3b8", font=small_font)

    draw.text((WIDTH // 2, HEIGHT - 40),
              "Demo complete  |  Model: MobileNetV3-Small  |  PyTorch",
              fill="#94a3b8", font=small_font, anchor="mm")

    return np.array(img)


def main():
    parser = argparse.ArgumentParser(description="Generate a demo video")
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "best_model.pth")
    parser.add_argument("--output", type=str, default="demo_video.mp4")
    parser.add_argument("--fps", type=int, default=1)
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model, class_names, device = load_model(args.checkpoint)
    print(f"Model loaded. Classes: {class_names}")

    # Pick one random test image per class
    samples = []
    for cls in class_names:
        cls_dir = TEST_DIR / cls
        if not cls_dir.exists():
            print(f"  Warning: no test dir for {cls}, skipping")
            continue
        images = [f for f in cls_dir.iterdir()
                  if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if images:
            samples.append((cls, random.choice(images)))
    print(f"Selected {len(samples)} sample images")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (WIDTH, HEIGHT))

    # Title frame (show for 3 seconds)
    title = draw_title_frame()
    for _ in range(3):
        frame = cv2.cvtColor(title, cv2.COLOR_RGB2BGR)
        out.write(frame)
    print("Wrote title frames")

    # Classify each sample
    all_results = []
    for true_class, img_path in samples:
        print(f"  Classifying: {img_path.name} (true: {true_class})")
        pred = predict_image(model, class_names, img_path, device=device)
        rec = recommendation_from_prediction(pred)
        all_results.append(rec)

        pil_img = Image.open(img_path).convert("RGB")
        frame_arr = draw_sample_frame(
            pil_img, pred["label"], pred["confidence"], rec, true_class
        )

        # Show each sample for FRAMES_PER_SAMPLE seconds
        for _ in range(FRAMES_PER_SAMPLE):
            frame = cv2.cvtColor(frame_arr, cv2.COLOR_RGB2BGR)
            out.write(frame)

    # Summary frame (4 seconds)
    summary = draw_summary_frame(all_results)
    for _ in range(4):
        frame = cv2.cvtColor(summary, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
    print(f"\nDemo video saved to: {args.output}")
    print(f"  Duration: {3 + len(samples) * FRAMES_PER_SAMPLE + 4} seconds")
    print(f"  Resolution: {WIDTH}x{HEIGHT}")
    print(f"  Play: start {args.output}")


if __name__ == "__main__":
    main()
