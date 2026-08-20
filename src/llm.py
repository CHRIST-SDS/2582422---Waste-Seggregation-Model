"""LLM layer: turns structured CV output into concise, human-friendly guidance.

Uses Ollama running locally (default http://localhost:11434).
The CV model answers "what is this?"; the LLM answers "what does it mean and
what should the user do?" using a short prompt built from the structured
prediction plus optional campus rules.
"""
import os

from openai import OpenAI

from .config import LLM_MAX_TOKENS, LLM_MODEL

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

DEFAULT_SYSTEM_PROMPT = (
    "You are a waste-sorting assistant for a campus smart waste-management system. "
    "You receive a structured prediction from a computer-vision model plus campus rules. "
    "Reply concisely in 2-3 short sentences. Always state the bin color and the exact "
    "action the user must take. If the prediction is low-confidence, ask a short "
    "clarifying question and tell the user how to verify the item."
)


def ollama_available() -> bool:
    """Check if an Ollama server is reachable."""
    try:
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        client.models.list()
        return True
    except Exception:
        return False


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
    model: str | None = None,
) -> str:
    """Call Ollama with the structured prediction. Returns the assistant message."""
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    response = client.chat.completions.create(
        model=model or LLM_MODEL,
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(prediction, context)},
        ],
        temperature=0.3,
        max_tokens=LLM_MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()
