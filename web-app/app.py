"""Simple Flask web app for milk tea nutrition detection."""

import os

from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "change-me-later"


@app.route("/", methods=["GET"])
def index():
    """Render the home page with the upload/scan form."""
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def result():
    """Handle image upload and display fake nutrition results."""
    image_file = request.files.get("image")

    if not image_file or image_file.filename == "":
        flash("Please select or scan a milk tea image first.")
        return redirect(url_for("index"))

    # Save the uploaded image (optional but useful if we want to show it later)
    uploads_dir = os.path.join(app.root_path, "static", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    image_path = os.path.join(uploads_dir, image_file.filename)
    image_file.save(image_path)

    # Placeholder for future real ML model / ML client call.
    nutrition = fake_nutrition_model(image_path)

    # Pass both nutrition info and image filename to template
    return render_template(
        "result.html",
        nutrition=nutrition,
        image_filename=image_file.filename,
    )


def fake_nutrition_model(image_path: str) -> dict:
    """
    Temporary fake model.

    Args:
        image_path: Path to the saved drink image (currently unused).

    Returns:
        A dictionary with dummy nutrition information.
    """
    # Mark the argument as intentionally unused for now
    _ = image_path

    return {
        "calories": 380,
        "sugar_grams": 38,
        "fat_grams": 8,
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6500, debug=True)
