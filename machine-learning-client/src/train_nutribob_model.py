"""Training script for the Nutribob image classification model.

This module:
- Loads training data from data/train using Keras image_dataset_from_directory
- Builds a MobileNetV2-based classifier
- Trains the model for a small number of epochs
- Saves the trained model and the detected class labels into the models/ folder
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "train")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "nutribob_model.h5")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.txt")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5


def load_datasets():
    """Load the training dataset from the data/train directory.

    Returns:
        A tuple (train_ds, val_ds, class_names) where:
            - train_ds: tf.data.Dataset containing training images and labels.
            - val_ds: currently None (no separate validation split is used).
            - class_names: list of class name strings inferred from subfolders.
    """
    print(f"Looking for training data in: {DATA_DIR}")

    if not os.path.exists(DATA_DIR):
        print("ERROR: data/train doesn't exist")
        sys.exit(1)

    subdirs = [
        d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))
    ]
    if not subdirs:
        print("ERROR: data/train has no subdirectories")
        sys.exit(1)

    print("Found class folders:", subdirs)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    class_names = train_ds.class_names
    print("Detected classes from dataset:", class_names)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(LABELS_PATH, "w", encoding="utf-8") as labels_file:
        for name in class_names:
            labels_file.write(name + "\n")
    print(f"Labels saved to {LABELS_PATH}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, None, class_names


def build_model(num_classes: int) -> tf.keras.Model:
    """Build a MobileNetV2-based image classification model.

    Args:
        num_classes: Number of output classes.

    Returns:
        A compiled Keras model ready for training.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )

    base_model.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def main() -> None:
    """Entry point for training the Nutribob model."""
    train_ds, val_ds, class_names = load_datasets()
    num_classes = len(class_names)

    model = build_model(num_classes)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )

    final_acc = history.history.get("accuracy", ["N/A"])[-1]
    print(f"Training finished. Final training accuracy: {final_acc}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Labels saved to {LABELS_PATH}")


if __name__ == "__main__":
    main()
