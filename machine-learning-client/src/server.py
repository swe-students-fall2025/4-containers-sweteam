"""Flask API for exposing the NutriBob ML pipeline to other services.

This service accepts an uploaded drink image, runs it through the
analyze_drink_image pipeline, and returns the resulting JSON payload.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from pipeline import analyze_drink_image

app = Flask(__name__)


def _save_upload_to_temp(file_storage) -> str:
    """Persist an uploaded file to a temporary path and return the path."""
    filename = secure_filename(file_storage.filename or "upload.jpg")
    _, ext = os.path.splitext(filename)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".jpg")
    file_storage.save(temp_file.name)
    temp_file.close()
    return temp_file.name


@app.route("/health", methods=["GET"])
def health() -> tuple[Any, int]:
    """Simple readiness endpoint for container orchestration."""
    return {"status": "ok"}, 200


@app.route("/analyze", methods=["POST"])
def analyze() -> tuple[Any, int]:
    """Accept an uploaded image, run the pipeline, and return the analysis."""
    image = request.files.get("image")
    if image is None or image.filename == "":
        return jsonify({"success": False, "error": "Image file missing"}), 400

    temp_path = _save_upload_to_temp(image)
    try:
        result = analyze_drink_image(temp_path)
    except FileNotFoundError:
        return jsonify({"success": False, "error": "Image could not be processed"}), 400
    except Exception as exc:  # pylint: disable=broad-except
        return (
            jsonify({"success": False, "error": f"Pipeline failed: {exc}"}),
            500,
        )
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    return jsonify(result), 200


def create_app() -> Flask:
    """Factory used by ASGI/WSGI servers."""
    return app


if __name__ == "__main__":
    port = int(os.environ.get("ML_SERVICE_PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
