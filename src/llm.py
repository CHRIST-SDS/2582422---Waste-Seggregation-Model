"""LLM layer: turns structured CV output into concise, human-friendly guidance.

Uses Ollama running locally (default http://localhost:11434).
The CV model answers "what is this?"; the LLM answers "what does it mean and
what should the user do?" using a short prompt built from the structured
prediction plus campus rules.
"""
import os

from openai import OpenAI

from .config import LLM_MAX_TOKENS, LLM_MODEL
from .recommend import BIN_RULES

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "30"))

SYSTEM_PROMPT = (
    "You are a campus waste-sorting assistant. "
    "Given a CV prediction and bin rules below, reply in 2-3 short sentences. "
    "Always state the bin color and the action the user must take. "
    f"{BIN_RULES}"
)


def ollama_available() -> bool:
    """Check if an Ollama server is reachable and has the required model."""
    try:
        client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",
            timeout=5,
        )
        models = client.models.list()
        available = [m.id for m in models.data]
        if LLM_MODEL not in available:
            print(f"Warning: '{LLM_MODEL}' not found. Available: {available}")
            print(f"Run: ollama pull {LLM_MODEL}")
            return False
        return True
    except Exception:
        return False


def build_user_prompt(prediction: dict) -> str:
    label = prediction.get("label", "unknown")
    confidence = prediction.get("confidence", 0.0)
    probas = prediction.get("probas", {})
    top = ", ".join(f"{k}: {v*100:.0f}%" for k, v in list(probas.items())[:3]) or "n/a"
    conf = "low" if prediction.get("low_confidence") else "high"
    return (
        f"Item identified as '{label}' ({conf} confidence, {confidence*100:.1f}%). "
        f"Top matches: {top}. "
        "What bin should the user use and what must they do first?"
    )


def generate_recommendation(
    prediction: dict,
    context: str = "",
    model: str | None = None,
) -> str:
    """Call Ollama with the structured prediction. Returns the assistant message."""
    client = OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
        timeout=LLM_TIMEOUT,
    )
    try:
        response = client.chat.completions.create(
            model=model or LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(prediction)},
            ],
            temperature=0.3,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        raise RuntimeError(
            f"LLM error: {exc}"
        ) from exc
