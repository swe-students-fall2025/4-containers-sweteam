import os
from classifier import classify_image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "train")

for class_name in sorted(os.listdir(DATA_DIR)):
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    for fname in os.listdir(class_dir):
        if fname.startswith("."):
            continue
        img_path = os.path.join(class_dir, fname)
        pred = classify_image(img_path)
        print(f"True folder: {class_name:20s} | File: {fname:30s} | Pred: {pred}")
