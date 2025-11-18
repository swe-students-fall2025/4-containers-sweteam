import os
import requests
from dotenv import load_dotenv

load_dotenv()

class NutritionAPIError(Exception):
    pass


def get_nutrition(query_text: str) -> dict:
    url = os.getenv("FOOD_API_URL")
    key = os.getenv("FOOD_API_KEY")

    if not url or not key:
        raise NutritionAPIError(
            "FOOD_API_URL or FOOD_API_KEY not set in environment variables, please check .env or docker-compose"
        )

    headers = {"X-Api-Key": key}
    params = {"query": query_text}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
    except requests.exceptions.RequestException as e:
        raise NutritionAPIError(f"Nutrition API call failed: {e}")

    if resp.status_code != 200:
        raise NutritionAPIError(
            f"Nutrition API returned error status code {resp.status_code}: {resp.text}"
        )

    try:
        data = resp.json()
    except ValueError:
        raise NutritionAPIError("Nutrition API returned invalid JSON")

    return data
