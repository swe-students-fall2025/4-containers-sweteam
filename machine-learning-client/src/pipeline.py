import os
from dotenv import load_dotenv

from classifier import classify_image
from recipe_mapper import get_recipe_for_label
from nutrition_api import get_nutrition, NutritionAPIError
from dotenv import load_dotenv

load_dotenv()

def analyze_drink_image(image_path: str) -> dict:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    label = classify_image(image_path)

    recipe_text = get_recipe_for_label(label)

    try:
        nutrition = get_nutrition(recipe_text)
    except NutritionAPIError as e:
        return {
            "success": False,
            "label": label,
            "recipe": recipe_text,
            "error": str(e),
        }

    result = {
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
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_image = os.path.join(
        BASE_DIR, "data", "train", "classic_milk_tea"
    )
    print("Please pass a real image path to analyze_drink_image() in your own test script.")

