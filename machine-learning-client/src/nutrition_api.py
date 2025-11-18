"""Helper utilities for calling the external nutrition API.

This module loads API configuration from environment variables and exposes
a simple function that wraps the HTTP request and normalizes errors.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()


class NutritionAPIError(Exception):
    """Custom exception type for nutrition API failures."""
    # No extra implementation needed; used for clearer error handling.


def get_nutrition(query_text: str) -> dict:
    """Fetch nutrition information for the given query text.

    Args:
        query_text: Free-text description of the food or drink.

    Returns:
        A dictionary parsed from the JSON response of the nutrition API.

    Raises:
        NutritionAPIError: If configuration is missing, the HTTP request fails,
        returns a non-200 status code, or the response body is not valid JSON.
    """
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
