"""Gradio web demo: upload image -> CV classifier -> LLM guidance -> bin recommendation.

Runs on localhost (default http://127.0.0.1:7860).

Usage:
    py app.py --checkpoint models/best_model.pth        # trained model + LLM (if key set)
    py app.py --checkpoint models/best_model.pth --no-llm
    py app.py --mock                                     # preview UI without a trained model

The LLM layer is used automatically when OPENAI_API_KEY is set (.env is auto-loaded).
Without a key, the app falls back to rule-based instructions so it still runs locally.
"""
import argparse
import html
import os
import random
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from src.config import CLASS_NAMES, MODELS_DIR
from src.llm import generate_recommendation
from src.predict import load_model, predict_image
from src.recommend import recommendation_from_prediction

load_dotenv()

PAGE_CSS = """
body {
  background: radial-gradient(1100px 550px at 15% -10%, #dbeafe 0%, transparent 60%),
              radial-gradient(1000px 500px at 95% 115%, #ede9fe 0%, transparent 55%),
              #f8fafc;
}
.gradio-container {
  max-width: 1100px !important;
  margin: 0 auto !important;
  background: transparent !important;
  font-family: 'Segoe UI', system-ui, sans-serif !important;
}
footer { display: none !important; }
"""

_CARD = (
    "background:#ffffff;border:1px solid #e2e8f0;border-radius:18px;padding:22px;"
    "color:#000000;font-family:'Segoe UI',system-ui,sans-serif;"
    "box-shadow:0 10px 30px rgba(15,23,42,0.08);"
)
_TEXT = "color:#000000;"
_EMPTY = (
    "text-align:center;color:#000000;padding:40px 10px;font-size:15px;"
    f"background:#ffffff;border:1px solid #e2e8f0;border-radius:18px;{_TEXT}"
)

EMPTY_HTML = (
    f'<div style="{_EMPTY}">Upload or capture an image to see the recommendation.</div>'
)


def _mock_probas(label: str, confidence: float) -> dict[str, float]:
    others = [c for c in CLASS_NAMES if c != label]
    random.shuffle(others)
    raw = {label: confidence, others[0]: confidence * 0.12, others[1]: confidence * 0.05}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


class WasteApp:
    def __init__(self, checkpoint: Path | None, mock: bool = False, use_llm: bool = True):
        self.mock = mock
        self.use_llm = use_llm
        self.model = None
        self.class_names = None
        self.device = None
        if mock:
            print("[mock] Simulating CV predictions (no checkpoint loaded).")
        else:
            self.model, self.class_names, self.device = load_model(checkpoint)

    def cv_predict(self, image):
        if self.mock:
            label = random.choice(CLASS_NAMES)
            confidence = random.uniform(0.70, 0.97)
            return {
                "label": label,
                "confidence": confidence,
                "probas": _mock_probas(label, confidence),
                "low_confidence": random.random() < 0.2,
            }
        return predict_image(self.model, self.class_names, image, device=self.device)

    def classify(self, image):
        """Returns a self-contained HTML card using only inline styles."""
        if image is None:
            return EMPTY_HTML

        prediction = self.cv_predict(image)
        rec = recommendation_from_prediction(prediction)
        label = rec["label"]
        confidence = rec["confidence"]
        hex_color = rec["hex_color"]

        llm_html = ""
        if self.use_llm:
            try:
                llm_html = (
                    f'<div style="margin-top:14px;padding:12px 14px;border-radius:12px;'
                    f'background:#eef2ff;border:1px solid #a5b4fc;font-size:14px;'
                    f'line-height:1.55;{_TEXT}">{html.escape(generate_recommendation(prediction))}</div>'
                )
            except Exception as exc:  # no API key / network issues -> fall back
                llm_html = (
                    f'<div style="margin-top:14px;padding:12px 14px;border-radius:12px;'
                    f'background:#eef2ff;border:1px solid #a5b4fc;font-size:14px;'
                    f'line-height:1.55;{_TEXT}">LLM unavailable: {html.escape(str(exc))}</div>'
                )

        warn_html = (
            f'<div style="margin-top:14px;padding:9px 12px;border-radius:10px;font-size:13px;'
            f'background:#fef3c7;border:1px solid #f59e0b;{_TEXT}">'
            "Low confidence prediction &mdash; please double-check the item before disposing.</div>"
            if rec["low_confidence"]
            else ""
        )

        section = (
            lambda text: f'<div style="font-size:11px;font-weight:700;letter-spacing:1.5px;'
            f"text-transform:uppercase;color:#000000;margin:16px 0 8px;\">{text}</div>"
        )

        return (
            f'<div style="{_CARD}">'
            f'<div style="display:flex;align-items:center;gap:14px;">'
            f'<div style="width:56px;height:56px;border-radius:16px;flex:none;'
            f'display:flex;align-items:center;justify-content:center;font-size:26px;'
            f'font-weight:800;color:#ffffff;background:{hex_color};'
            f'box-shadow:0 6px 20px rgba(15,23,42,0.25);">{html.escape(label[0].upper())}</div>'
            f'<div><div style="font-size:24px;font-weight:800;line-height:1.1;color:{hex_color};">'
            f"{html.escape(rec['bin'])} BIN</div>"
            f'<div style="font-size:13px;margin-top:2px;{_TEXT}">{html.escape(rec["stream"])}</div></div>'
            f'<div style="margin-left:auto;text-align:right;">'
            f'<div style="font-size:28px;font-weight:800;line-height:1;color:{hex_color};">'
            f"{confidence * 100:.0f}%</div>"
            f'<div style="font-size:12px;{_TEXT}">confidence</div></div></div>'
            f'<div style="height:8px;border-radius:99px;background:#e2e8f0;overflow:hidden;margin-top:16px;">'
            f'<div style="height:100%;width:{confidence * 100:.0f}%;border-radius:99px;background:{hex_color};"></div>'
            f"</div>"
            f"{section('Detected')}"
            f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0;">'
            f'<span style="width:84px;font-size:13px;text-transform:capitalize;{_TEXT}">'
            f"{html.escape(label)}</span>"
            f'<div style="flex:1;height:8px;border-radius:99px;background:#e2e8f0;overflow:hidden;">'
            f'<div style="height:100%;width:{confidence * 100:.0f}%;border-radius:99px;background:{hex_color};"></div>'
            f"</div>"
            f'<span style="width:44px;text-align:right;font-size:12px;{_TEXT}">'
            f"{confidence * 100:.0f}%</span></div>"
            f"{section('Instructions')}"
            f'<div style="font-size:15px;line-height:1.55;{_TEXT}">{html.escape(rec["instructions"])}</div>'
            f"{warn_html}"
            f"{llm_html}</div>"
        )


def main():
    parser = argparse.ArgumentParser(description="Launch the smart waste segregation web demo")
    parser.add_argument("--checkpoint", type=Path, default=None, help="path to best_model.pth")
    parser.add_argument("--mock", action="store_true", help="simulate CV predictions (UI preview)")
    parser.add_argument("--no-llm", action="store_true", help="skip the LLM layer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public share link")
    args = parser.parse_args()

    checkpoint = args.checkpoint or (MODELS_DIR / "best_model.pth")
    if not args.mock and not checkpoint.is_file():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint}\n"
            "Train first with 'py -m src.train', or use --mock to preview the UI."
        )

    use_llm = not args.no_llm
    if use_llm and not os.environ.get("OPENAI_API_KEY"):
        print("Note: OPENAI_API_KEY not set — LLM layer disabled, using rule-based guidance.")
        use_llm = False

    app = WasteApp(checkpoint, mock=args.mock, use_llm=use_llm)

    with gr.Blocks(title="Smart Waste Segregation") as demo:
        gr.HTML(
            '<div style="font-size:34px;font-weight:800;color:#000000;">AI Waste Segregation</div>'
            '<div style="color:#000000;font-size:15px;margin-bottom:18px;">Camera &rarr; '
            "Computer vision &rarr; LLM guidance &rarr; color-coded bin recommendation</div>"
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Show the item",
                    sources=["upload", "webcam"],
                    height=380,
                )
                classify_btn = gr.Button("Classify", variant="primary", size="lg")
            with gr.Column(scale=1):
                result = gr.HTML(EMPTY_HTML, elem_id="wc-result")

        image_input.change(app.classify, inputs=image_input, outputs=result)
        classify_btn.click(app.classify, inputs=image_input, outputs=result)

    demo.launch(server_name=args.host, server_port=args.port, share=args.share, css=PAGE_CSS)


if __name__ == "__main__":
    main()
