from flask import Flask, render_template, request, redirect, url_for, flash
import os

app = Flask(__name__)
app.secret_key = "change-me-later"  # needed for flash messages

# For now we just use an in-memory list instead of MongoDB
DRINK_HISTORY = []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scan", methods=["GET", "POST"])
def scan():
    if request.method == "POST":
        # Get form fields
        size = request.form.get("size")
        sweetness = request.form.get("sweetness")
        toppings = request.form.get("toppings")
        image_file = request.files.get("image")

        if not image_file or image_file.filename == "":
            flash("Please upload an image of your milk tea.")
            return redirect(url_for("scan"))

        # Save uploaded image to static/uploads
        uploads_dir = os.path.join(app.root_path, "static", "uploads")
        os.makedirs(uploads_dir, exist_ok=True)

        # Very naive filename handling (OK for assignment demo)
        image_path = os.path.join(uploads_dir, image_file.filename)
        image_file.save(image_path)

        # TODO: call ML client here to get real predictions
        # For now, use dummy "model"
        calories, sugar_grams = fake_nutrition_model(size, sweetness, toppings)

        # Save a record (later this will go to MongoDB)
        record = {
            "size": size,
            "sweetness": sweetness,
            "toppings": toppings,
            "image_filename": image_file.filename,
            "calories": calories,
            "sugar_grams": sugar_grams,
        }
        DRINK_HISTORY.append(record)

        return render_template("result.html", record=record)

    # GET request
    return render_template("scan.html")


@app.route("/history")
def history():
    # Later we’ll load from MongoDB instead of DRINK_HISTORY
    return render_template("history.html", drinks=DRINK_HISTORY)


def fake_nutrition_model(size, sweetness, toppings):
    """
    Temporary 'model' so the app works before we build the real ML client.
    """
    base_calories = {"small": 250, "medium": 350, "large": 450}
    base_sugar = {"small": 25, "medium": 35, "large": 45}

    size_key = size or "medium"
    cals = base_calories.get(size_key, 350)
    sugar = base_sugar.get(size_key, 35)

    if sweetness == "50":
        cals *= 0.7
        sugar *= 0.7
    elif sweetness == "30":
        cals *= 0.5
        sugar *= 0.5

    if toppings:
        toppings_list = [t.strip().lower() for t in toppings.split(",")]
        for t in toppings_list:
            if "boba" in t or "pearl" in t:
                cals += 80
                sugar += 8
            elif "pudding" in t or "jelly" in t:
                cals += 50
                sugar += 5

    return int(cals), int(sugar)


if __name__ == "__main__":
    # Run on port 6500 to match your previous instructions
    app.run(host="0.0.0.0", port=6500, debug=True)
