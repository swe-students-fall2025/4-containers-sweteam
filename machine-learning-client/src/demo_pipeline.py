"""Small demo script to run the Nutribob analysis pipeline on a sample image."""

import os
from pipeline import analyze_drink_image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_IMAGE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "train",
    "fruit_tea",
    "DCREAM-CAT-02-FRUITTEA.jpg",
)

if __name__ == "__main__":
    result = analyze_drink_image(TEST_IMAGE_PATH)
    print("Analysis result:")
    print(result)
