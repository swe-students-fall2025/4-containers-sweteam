"""End-to-end image analysis pipeline for the Nutribob project.

This module defines analyze_drink_image(), which performs:
1. Image classification using the ML model.
2. Recipe lookup based on predicted label.
3. Nutrition API enrichment to fetch calorie estimates.
"""

import os
from dotenv import load_dotenv
from classifier import classify_image
from recipe_mapper import get_recipe_for_label
from nutrition_api import get_nutrition, NutritionAPIError

load_dotenv()


def analyze_drink_image(image_path: str) -> dict:
    """Analyze a drink image and return label, recipe, and nutrition data.

    Args:
        image_path: Path to the drink image.

    Returns:
        A dictionary containing:
            - success (bool)
            - label (str)
            - recipe (str)
            - nutrition_raw (dict, optional)
            - summary (dict, optional)
            - error (str, optional)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    label = classify_image(image_path)
    recipe_text = get_recipe_for_label(label)

    try:
        nutrition = get_nutrition(recipe_text)
    except NutritionAPIError as exc:
        return {
            "success": False,
            "label": label,
            "recipe": recipe_text,
            "error": str(exc),
        }

    result: dict = {
        "success": True,
        "label": label,
        "recipe": recipe_text,
        "nutrition_raw": nutrition,
    }

    items = nutrition.get("items", [])
    if items:
        total_calories = sum(item.get("calories", 0) for item in items)
        result["summary"] = {
            "total_calories": total_calories,
            "items_count": len(items),
        }

    return result


if __name__ == "__main__":
    print(
        "This module provides analyze_drink_image(). "
        "Use it by importing and calling analyze_drink_image('path/to/image.jpg')."
    )
