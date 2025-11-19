DRINK_RECIPES = {
    "classic_milk_tea": "black tea, whole milk, sugar syrup, tapioca pearls",
    "taro_milk_tea": "taro powder, whole milk, sugar syrup, tapioca pearls",
    "fruit_tea": "green tea, fruit juice, fresh fruit, sugar syrup",
    "matcha_milk_tea": "matcha powder, whole milk, sugar syrup",
    "brown_sugar_milk_tea": "whole milk, brown sugar syrup, tapioca pearls",
    "unknown": "milk tea",
}


def get_recipe_for_label(label: str) -> str:
    return DRINK_RECIPES.get(label, DRINK_RECIPES["unknown"])
