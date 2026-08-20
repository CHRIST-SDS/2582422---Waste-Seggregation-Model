"""LLM layer: turns structured CV output into concise, human-friendly guidance.

Supports two backends:
  - Ollama (local, default) — runs Llama 3.2 3B via localhost:11434
  - OpenAI — runs GPT-4o-mini via api.openai.com (requires OPENAI_API_KEY)

If neither backend is available, callers should fall back to rule-based guidance.
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


def _ollama_available() -> bool:
    """Check if an Ollama server is reachable."""
    try:
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        client.models.list()
        return True
    except Exception:
        return False


def _openai_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _client():
    """Pick the best available client: Ollama first, then OpenAI."""
    if _ollama_available():
        return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama"), "ollama"
    if _openai_available():
        return OpenAI(), "openai"
    raise RuntimeError(
        "No LLM backend available. Install and start Ollama (ollama.com/download) "
        "and pull a model (ollama pull llama3.2:3b), or set OPENAI_API_KEY in .env."
    )


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
    """Call the best available LLM with the structured prediction. Returns the assistant message."""
    client, backend = _client()

    if backend == "ollama":
        model = model or LLM_MODEL
    else:
        model = model or LLM_MODEL

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
