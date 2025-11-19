"""Image classification utilities for the Nutribob project.

This module loads the pre-trained Keras model and provides helper functions
to preprocess drink images and run predictions. When the TensorFlow model
or weights are unavailable (common on developer machines), the classifier
falls back to a simple stub prediction so the service stays usable.
"""

import os
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:  # TensorFlow not installed (e.g., CI or lightweight dev env)
    tf = None
    load_model = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "nutribob_model.h5")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.txt")

IMG_SIZE = (224, 224)

LABELS: list[str] = []
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        LABELS = [line.strip() for line in f.readlines() if line.strip()]
if not LABELS:
    LABELS = ["Milk tea"]

model = None
MODEL_AVAILABLE = False

if load_model is not None and os.path.exists(MODEL_PATH):
    try:
        print(f"[classifier] Loading model from {MODEL_PATH} ...")
        model = load_model(MODEL_PATH)
        MODEL_AVAILABLE = True
        print("[classifier] Model loaded.")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[classifier] Failed to load TensorFlow model: {exc}")
else:
    if load_model is None:
        print("[classifier] TensorFlow not available; using stub classifier.")
    else:
        print(
            f"[classifier] Model file not found at {MODEL_PATH}; "
            "using stub classifier."
        )


def preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess an image for classification.

    The image is resized, converted to an array, normalized to [0, 1],
    and expanded with a batch dimension.

    Args:
        image_path: Path to the image file.

    Returns:
        A NumPy array of shape (1, height, width, channels) ready for the model.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not MODEL_AVAILABLE or tf is None:
        raise RuntimeError("TensorFlow model is not available for preprocessing.")

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def classify_image(image_path: str) -> str:
    """Run a prediction on the given image and return the predicted label.

    Args:
        image_path: Path to the image file to classify.

    Returns:
        The predicted class label as a string. Returns "unknown" if the
        predicted class id is out of range of the loaded labels.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not MODEL_AVAILABLE or model is None:
        # Stub prediction so the ML API can function without heavy model files.
        print("[classifier] MODEL_AVAILABLE is False; returning stub label.")
        return LABELS[0]

    img_array = preprocess_image(image_path)

    predictions = model.predict(img_array)
    class_id = int(np.argmax(predictions, axis=1)[0])

    if class_id < 0 or class_id >= len(LABELS):
        return "unknown"

    label = LABELS[class_id]
    print(f"[classifier] Predicted class id: {class_id}, label: {label}")
    return label


if __name__ == "__main__":
    # This module is intended to be imported by other code.
    print("[classifier] This module is meant to be imported, not run directly.")
