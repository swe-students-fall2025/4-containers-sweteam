"""Training script for the NutriBob MobileNetV2 classification model."""

import os
import sys
from typing import Dict, List, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "train")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "nutribob_model.h5")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.txt")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
SEED = 42

AUTOTUNE = tf.data.AUTOTUNE


def _check_data_dir() -> List[str]:
    """Ensure the training data directory exists and contains class folders."""
    print(f"Looking for training data in: {DATA_DIR}")

    if not os.path.exists(DATA_DIR):
        print("ERROR: data/train doesn't exist")
        sys.exit(1)

    subdirs = [
        d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))
    ]
    subdirs.sort()

    if not subdirs:
        print("ERROR: data/train has no subdirectories (class folders).")
        sys.exit(1)

    print("Found class folders:", subdirs)
    return subdirs


def load_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """Load and split the training data into train/validation datasets."""
    _check_data_dir()

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_split=0.2,
        subset="training",
        seed=SEED,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
    )

    class_names = train_ds.class_names
    print("Detected classes from dataset:", class_names)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(LABELS_PATH, "w", encoding="utf-8") as label_file:
        for name in class_names:
            label_file.write(name + "\n")
    print(f"Labels saved to {LABELS_PATH}")

    def prepare(ds, training: bool) -> tf.data.Dataset:
        if training:
            ds = ds.shuffle(buffer_size=1000, seed=SEED)
        return ds.prefetch(buffer_size=AUTOTUNE)

    train_ds = prepare(train_ds, training=True)
    val_ds = prepare(val_ds, training=False)

    return train_ds, val_ds, class_names


def compute_class_weights(class_names: List[str]) -> Dict[int, float]:
    """Compute a weighting for each class to balance the training data."""
    counts: List[int] = []
    for _, name in enumerate(class_names):
        folder = os.path.join(DATA_DIR, name)
        num_files = len(
            [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        )
        print(f"Class '{name}' has {num_files} images.")
        if num_files == 0:
            print(
                f"WARNING: class '{name}' has 0 images; "
                "this may cause training issues."
            )
        counts.append(max(num_files, 1))

    total = sum(counts)
    num_classes = len(class_names)
    class_weights = {i: total / (num_classes * count) for i, count in enumerate(counts)}

    print("Computed class weights:", class_weights)
    return class_weights


def build_model(num_classes: int) -> tf.keras.Model:
    """Construct and compile the MobileNetV2 transfer-learning model."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )

    base_model.trainable = False

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    inputs = layers.Input(shape=IMG_SIZE + (3,), name="image_input")

    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="nutribob_mobilenetv2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def main() -> None:
    """Entry point that orchestrates dataset loading, training, and saving."""
    print("=== Loading datasets ===")
    train_ds, val_ds, class_names = load_datasets()
    num_classes = len(class_names)

    print("\n=== Building model ===")
    model = build_model(num_classes)

    print("\n=== Computing class weights ===")
    class_weights = compute_class_weights(class_names)

    print("\n=== Start training ===")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
    )

    final_acc = history.history.get("accuracy", ["N/A"])[-1]
    final_val_acc = history.history.get("val_accuracy", ["N/A"])[-1]
    print("\nTraining finished.")
    print(f"Final training accuracy: {final_acc}")
    print(f"Final validation accuracy: {final_val_acc}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Labels saved to {LABELS_PATH}")


if __name__ == "__main__":
    main()
