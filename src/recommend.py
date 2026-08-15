"""Bin mapping and disposal recommendations for each waste class."""
from dataclasses import asdict, dataclass

from .config import LOW_CONFIDENCE_THRESHOLD


@dataclass
class BinInfo:
    name: str
    hex_color: str
    stream: str
    instructions: str


BINS = {
    "organic": BinInfo(
        "GREEN", "#4CAF50", "Biodegradable / Organic waste",
        "Place food and organic waste in the GREEN bin for composting.",
    ),
    "cardboard": BinInfo(
        "BLUE", "#2196F3", "Recyclable / Dry waste",
        "Empty and flatten the cardboard, then place it in the BLUE recycling bin.",
    ),
    "paper": BinInfo(
        "BLUE", "#2196F3", "Recyclable / Dry waste",
        "Keep paper dry and clean, then place it in the BLUE recycling bin.",
    ),
    "plastic": BinInfo(
        "BLUE", "#2196F3", "Recyclable / Dry waste",
        "Empty and rinse the plastic item, then place it in the BLUE recycling bin.",
    ),
    "metal": BinInfo(
        "BLUE", "#2196F3", "Recyclable / Dry waste",
        "Rinse cans and tins, then place them in the BLUE recycling bin.",
    ),
    "glass": BinInfo(
        "RED", "#EF4444", "Non-recyclable / Special waste",
        "Glass items are not accepted as recyclables here (e.g. lab bottles). "
        "Place them in the RED bin for proper disposal.",
    ),
    "trash": BinInfo(
        "RED", "#EF4444", "Non-recyclable / Residual",
        "Place this item in the RED residual-waste bin. Do not mix it with recyclables.",
    ),
}


def bin_for_class(class_name: str) -> BinInfo | None:
    return BINS.get(class_name)


def recommendation_from_prediction(prediction: dict) -> dict:
    """Map a structured CV prediction to a color-coded bin recommendation.

    prediction keys: label, confidence, probas, low_confidence.
    """
    label = prediction.get("label", "unknown")
    confidence = prediction.get("confidence", 0.0)
    bin_info = bin_for_class(label)

    if bin_info is None:
        rec = {
            "bin": "GRAY",
            "hex_color": "#9E9E9E",
            "stream": "Unknown category",
            "instructions": f"Unable to map '{label}' to a bin. Please verify the item.",
        }
    else:
        rec = asdict(bin_info)
        rec["bin"] = bin_info.name

    rec["label"] = label
    rec["confidence"] = confidence
    rec["low_confidence"] = prediction.get("low_confidence", confidence < LOW_CONFIDENCE_THRESHOLD)
    rec["probas"] = prediction.get("probas", {})

    if rec["low_confidence"]:
        rec["instructions"] += " Prediction confidence is low — please double-check before disposing."
    return rec
