"""Tests for analyze_drink_image() defined in src/pipeline.py"""

import os
import sys
from pathlib import Path

import pytest

from pipeline import analyze_drink_image
from nutrition_api import NutritionAPIError

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def test_missing_image_raises_file_not_found(tmp_path):
    fake_path = tmp_path / "non_existent_image.jpg"

    with pytest.raises(FileNotFoundError):
        analyze_drink_image(str(fake_path))


def test_successful_analysis_builds_summary(monkeypatch, tmp_path):
    img_path = tmp_path / "dummy.jpg"
    img_path.write_bytes(b"fake image data")

    def fake_classify_image(path: str) -> str:
        assert os.path.exists(path)
        return "matcha_milk_tea"

    def fake_get_recipe_for_label(label: str) -> str:
        assert label == "matcha_milk_tea"
        return "1 cup milk tea with tapioca pearls"

    def fake_get_nutrition(recipe_text: str) -> dict:
        assert "milk tea" in recipe_text
        return {
            "items": [
                {"name": "milk", "calories": 100},
                {"name": "tea", "calories": 10},
                {"name": "sugar", "calories": 50},
            ]
        }

    monkeypatch.setattr("pipeline.classify_image", fake_classify_image)
    monkeypatch.setattr("pipeline.get_recipe_for_label", fake_get_recipe_for_label)
    monkeypatch.setattr("pipeline.get_nutrition", fake_get_nutrition)

    result = analyze_drink_image(str(img_path))

    assert result["success"] is True
    assert result["label"] == "matcha_milk_tea"
    assert "recipe" in result
    assert "nutrition_raw" in result

    summary = result.get("summary")
    assert summary is not None
    assert summary["total_calories"] == 160
    assert summary["items_count"] == 3


def test_nutrition_api_error_returns_failure(monkeypatch, tmp_path):
    img_path = tmp_path / "dummy.jpg"
    img_path.write_bytes(b"fake image data")

    def fake_classify_image(path: str) -> str:
        return "taro_milk_tea"

    def fake_get_recipe_for_label(label: str) -> str:
        return f"fake recipe for {label}"

    def fake_get_nutrition(recipe_text: str):
        raise NutritionAPIError("API quota exceeded")

    monkeypatch.setattr("pipeline.classify_image", fake_classify_image)
    monkeypatch.setattr("pipeline.get_recipe_for_label", fake_get_recipe_for_label)
    monkeypatch.setattr("pipeline.get_nutrition", fake_get_nutrition)

    result = analyze_drink_image(str(img_path))

    assert result["success"] is False
    assert result["label"] == "taro_milk_tea"
    assert "recipe" in result
    assert "error" in result
    assert "API quota exceeded" in result["error"]
