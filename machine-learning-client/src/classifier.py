"""Core modules for the Nutribob machine learning client."""

import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "nutribob_model.h5"
LABELS_PATH = MODELS_DIR / "labels.txt"

IMG_SIZE = (224, 224)

_MODEL = None
_LABELS: List[str] = []


def _load_labels() -> List[str]:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _ensure_model_loaded():
    # pylint: disable=global-statement
    global _MODEL, _LABELS
    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        if not LABELS_PATH.exists():
            raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

        print(f"[classifier] Loading model from {MODEL_PATH} ...")
        _MODEL = load_model(MODEL_PATH)
        _LABELS = _load_labels()
        print(f"[classifier] Model loaded. Labels: {_LABELS}")


def _preprocess_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)

    arr = np.array(img).astype("float32")
    return np.expand_dims(arr, axis=0)


def classify_image(image_path: str) -> str:
    """Classify the drink in the given image and return its label."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    _ensure_model_loaded()

    x = _preprocess_image(image_path)
    preds = _MODEL.predict(x, verbose=0)[0]
    class_id = int(np.argmax(preds))
    label = _LABELS[class_id]

    print(
        f"[classifier] Predicted class id: {class_id}, "
        f"label: {label}, probs: {np.round(preds, 4)}"
    )
    return label
