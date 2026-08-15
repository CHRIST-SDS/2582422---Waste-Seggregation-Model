"""LLM layer: turns structured CV output into concise, human-friendly guidance (OpenAI).

The CV model answers "what is this?"; the LLM answers "what does it mean and
what should the user do?" using a short prompt built from the structured
prediction plus optional campus rules.
"""
import os

from openai import OpenAI

from .config import LLM_MAX_TOKENS, LLM_MODEL

DEFAULT_SYSTEM_PROMPT = (
    "You are a waste-sorting assistant for a campus smart waste-management system. "
    "You receive a structured prediction from a computer-vision model plus campus rules. "
    "Reply concisely in 2-3 short sentences. Always state the bin color and the exact "
    "action the user must take. If the prediction is low-confidence, ask a short "
    "clarifying question and tell the user how to verify the item."
)


def _api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key, "
            "then re-run (the app auto-loads .env)."
        )
    return key


def build_user_prompt(prediction: dict, context: str = "") -> str:
    label = prediction.get("label", "unknown")
    confidence = prediction.get("confidence", 0.0)
    probas = prediction.get("probas", {})
    top = ", ".join(f"{k}: {v * 100:.0f}%" for k, v in list(probas.items())[:3]) or "n/a"
    ambiguity = "low confidence" if prediction.get("low_confidence") else "high confidence"

    lines = [
        f"The computer-vision model identified '{label}' with {ambiguity} "
        f"(confidence {confidence * 100:.1f}%).",
        f"Top probabilities: {top}.",
    ]
    if context:
        lines.append(f"Campus rules: {context}")
    lines.append("What bin should the user use and what must they do first?")
    return "\n".join(lines)


def generate_recommendation(
    prediction: dict,
    context: str = "",
    api_key: str | None = None,
    model: str = LLM_MODEL,
) -> str:
    """Call OpenAI with the structured prediction. Returns the assistant message."""
    client = OpenAI(api_key=api_key or _api_key())
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(prediction, context)},
        ],
        temperature=0.3,
        max_tokens=LLM_MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()
