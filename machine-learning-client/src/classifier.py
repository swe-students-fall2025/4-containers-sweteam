"""Image classification helpers for the NutriBob machine-learning client."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency during linting
    np = None  # type: ignore

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

try:
    from tensorflow.keras.models import load_model
except ImportError:  # pragma: no cover
    load_model = None  # type: ignore

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "nutribob_model.h5"
LABELS_PATH = MODELS_DIR / "labels.txt"

IMG_SIZE = (224, 224)

_STATE: dict[str, Any] = {"model": None, "labels": [], "model_available": False}


def _load_labels() -> List[str]:
    """Read the label file from disk or provide a single fallback label."""
    if LABELS_PATH.exists():
        with open(LABELS_PATH, "r", encoding="utf-8") as label_file:
            labels = [line.strip() for line in label_file if line.strip()]
            if labels:
                return labels
    return ["Milk tea"]


def _ensure_model_loaded() -> None:
    """Load the TensorFlow model if the environment supports it."""
    if _STATE["model"] is not None:
        return

    if load_model is None or np is None or Image is None:
        print("[classifier] TensorFlow stack not available; using stub mode.")
        _STATE.update(
            {"model": None, "labels": _load_labels(), "model_available": False}
        )
        return

    if not MODEL_PATH.exists():
        print(f"[classifier] Model file not found at {MODEL_PATH}; using stub.")
        _STATE.update(
            {"model": None, "labels": _load_labels(), "model_available": False}
        )
        return

    print(f"[classifier] Loading model from {MODEL_PATH} ...")
    model = load_model(MODEL_PATH)
    labels = _load_labels()
    _STATE.update({"model": model, "labels": labels, "model_available": True})
    print("[classifier] Model loaded.")


def _preprocess_image(image_path: str) -> Any:
    """Load and preprocess an image for the TensorFlow classifier."""
    if Image is None or np is None:
        raise RuntimeError("TensorFlow preprocessing requires Pillow and NumPy.")

    with Image.open(image_path).convert("RGB") as img:
        resized = img.resize(IMG_SIZE)
        arr = np.array(resized, dtype="float32")
        return np.expand_dims(arr, axis=0)


def classify_image(image_path: str) -> str:
    """Return the predicted label for the supplied image path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    _ensure_model_loaded()
    labels: List[str] = _STATE["labels"] or ["Milk tea"]

    if not _STATE["model_available"] or _STATE["model"] is None:
        print("[classifier] MODEL_AVAILABLE False; returning stub prediction.")
        return labels[0]

    if np is None:
        raise RuntimeError("NumPy is required for TensorFlow predictions.")

    model = _STATE["model"]
    inputs = _preprocess_image(image_path)
    preds = model.predict(inputs, verbose=0)[0]
    class_id = int(np.argmax(preds))
    label = labels[class_id] if 0 <= class_id < len(labels) else labels[0]

    print(
        f"[classifier] Predicted class id: {class_id}, "
        f"label: {label}, probs: {np.round(preds, 4)}"
    )
    return label
