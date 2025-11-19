"""Recipe mapping utilities for Nutribob.

This module maps predicted drink labels to ingredient lists that can be
passed to the nutrition API.
"""

DRINK_RECIPES = {
    "classic_milk_tea": "black tea, whole milk, sugar syrup, tapioca pearls",
    "taro_milk_tea": "taro powder, whole milk, sugar syrup, tapioca pearls",
    "fruit_tea": "green tea, fruit juice, fresh fruit, sugar syrup",
    "matcha_milk_tea": "matcha powder, whole milk, sugar syrup",
    "brown_sugar_milk_tea": "whole milk, brown sugar syrup, tapioca pearls",
    "unknown": "milk tea",
}


def get_recipe_for_label(label: str) -> str:
    """Return the ingredient list associated with a drink label.

    Args:
        label: Classification label predicted by the ML model.

    Returns:
        A string describing ingredients for the drink. Defaults to "milk tea"
        if the label is unrecognized.
    """
    return DRINK_RECIPES.get(label, DRINK_RECIPES["unknown"])
