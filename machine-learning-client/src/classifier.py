"""Image classification utilities for the Nutribob project.

This module loads the pre-trained Keras model and provides helper functions
to preprocess drink images and run predictions.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "nutribob_model.h5")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.txt")

IMG_SIZE = (224, 224)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

print(f"[classifier] Loading model from {MODEL_PATH} ...")
model = load_model(MODEL_PATH)
print("[classifier] Model loaded.")

if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    LABELS = [line.strip() for line in f.readlines()]


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
