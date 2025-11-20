"""Wrapper utilities for calling the external nutrition API service."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()


class NutritionAPIError(Exception):
    """Custom exception type for nutrition API failures."""

    # No extra implementation needed; used for clearer error handling.


def get_nutrition(query_text: str) -> dict:
    """Call the configured nutrition API with the provided query text."""
    url = os.getenv("FOOD_API_URL")
    key = os.getenv("FOOD_API_KEY")

    if not url or not key:
        raise NutritionAPIError(
            "FOOD_API_URL or FOOD_API_KEY not set in environment variables; "
            "please check .env or docker-compose."
        )

    headers = {"X-Api-Key": key}
    params = {"query": query_text}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise NutritionAPIError(f"Nutrition API call failed: {exc}") from exc

    try:
        data = resp.json()
    except ValueError as exc:
        raise NutritionAPIError("Nutrition API returned invalid JSON") from exc

    return data
