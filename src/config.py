"""Shared project configuration (paths, constants, defaults)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "dataset-resized"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = ROOT / "models"

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DEFAULT_ARCH = "resnet18"
DEFAULT_EPOCHS = 25
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3

# Predictions below this confidence are treated as ambiguous (LLM asks to verify).
LOW_CONFIDENCE_THRESHOLD = 0.65

# LLM
LLM_MODEL = "gpt-4o-mini"
LLM_MAX_TOKENS = 200
