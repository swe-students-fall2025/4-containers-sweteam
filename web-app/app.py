from flask import Flask, render_template, request, redirect, url_for, flash
import os

app = Flask(__name__)
app.secret_key = "change-me-later"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def result():
    image_file = request.files.get("image")

    if not image_file or image_file.filename == "":
        flash("Please select or scan a milk tea image first.")
        return redirect(url_for("index"))

    # Save the uploaded image (optional but nice if you want to show it)
    uploads_dir = os.path.join(app.root_path, "static", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    image_path = os.path.join(uploads_dir, image_file.filename)
    image_file.save(image_path)

    # TODO: replace this with a real ML model / ML client call
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
    Later, you'll call your ML container instead of this.
    """
    # For now, just return some dummy numbers
    return {
        "calories": 380,
        "sugar_grams": 38,
        "fat_grams": 8,
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)
