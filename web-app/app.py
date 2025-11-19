"""Flask web application for NutriBob milk tea nutrition scanning.

This module sets up the web UI, Google SSO authentication, and routes for
uploading images to be analyzed by the (currently fake) nutrition model.
"""

import os
import io
from datetime import datetime
from functools import wraps
import certifi
from bson import ObjectId

from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
    send_file,
    abort,
)
from pymongo.mongo_client import MongoClient
from bson.binary import Binary

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me-later")

mongo_client: MongoClient | None = None
mongo_db = None

mongo_uri = os.environ.get("MONGODB_URI")

if mongo_uri:
    try:
        mongo_client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=5000,
            tlsCAFile=certifi.where(),
        )
        mongo_db = mongo_client["nutribob"]
        print("Connected to MongoDB successfully.")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"MongoDB connection failed: {exc}")
        mongo_client = None
        mongo_db = None

oauth = OAuth(app)
oauth.register(
    name="google",
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    api_base_url="https://www.googleapis.com/oauth2/v3/",
    client_kwargs={"scope": "openid email profile"},
)


def login_required(view):
    """Decorator that redirects anonymous users to the login page."""

    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


@app.route("/", methods=["GET"])
def home():
    """Public landing route. Redirect logged-in users to the scanner."""
    if "user" in session:
        return redirect(url_for("index"))
    return redirect(url_for("login"))


@app.route("/scan", methods=["GET"])
@login_required
def index():
    """Render the home page with the upload/scan form."""
    return render_template("index.html", user=session.get("user"))


@app.route("/scan", methods=["POST"])
@login_required
def scan():
    """Handle image upload from the scan page and show results."""
    image_file1 = request.files.get("image1")
    image_file2 = request.files.get("image2")
    image_file = image_file2 if image_file1.filename == "" else image_file1

    if not image_file1 and not image_file2:
        flash("Please select or scan a milk tea image first.")
        return redirect(url_for("index"))

    uploads_dir = os.path.join(app.root_path, "static", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    image_path = os.path.join(uploads_dir, image_file.filename)
    image_file.save(image_path)

    nutrition = fake_nutrition_model(image_path)

    user = session.get("user")
    if mongo_db is not None and user:
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            max_size_bytes = 16 * 1024 * 1024  # 16 MB
            if len(image_bytes) > max_size_bytes:
                flash("Image too large. Please upload an image under 16MB.")
                return redirect(url_for("index"))

            mongo_db.scans.insert_one(
                {
                    "user_id": user.get("id"),
                    "user_email": user.get("email"),
                    "image_filename": image_file.filename,
                    "image_content_type": image_file.mimetype,
                    "image_data": Binary(image_bytes),
                    "nutrition": nutrition,
                    "created_at": datetime.utcnow(),
                }
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Mongo insert failed: {exc}")

    return render_template(
        "result.html",
        nutrition=nutrition,
        image_filename=image_file.filename,
        user=user,
    )


@app.route("/history", methods=["GET"])
@login_required
def history():
    """Show a list of past scans for the logged-in user.

    If the database is unavailable, this falls back to a helpful message
    instead of failing.
    """
    user = session.get("user")
    scans = []

    if mongo_db is not None and user:
        try:
            cursor = (
                mongo_db.scans.find({"user_id": user.get("id")})
                .sort("created_at", -1)
                .limit(50)
            )
            for doc in cursor:
                scans.append(
                    {
                        "id": str(doc.get("_id")),
                        "drink": doc.get("drink_name", "Milk tea"),
                        "calories": doc.get("nutrition", {}).get("calories"),
                    }
                )
        except Exception:  # pylint: disable=broad-except
            scans = []
    return render_template(
        "history.html",
        user=user,
        scans=scans,
    )


@app.route("/image/<scan_id>")
@login_required
def image(scan_id):
    """Serve the stored image for a given scan document."""

    if mongo_db is None:
        abort(503)

    user = session.get("user")
    try:
        doc = mongo_db.scans.find_one(
            {"_id": ObjectId(scan_id), "user_id": user.get("id")}
        )
    except Exception:  # pylint: disable=broad-except
        doc = None

    if not doc or "image_data" not in doc:
        abort(404)

    return send_file(
        io.BytesIO(doc["image_data"]),
        mimetype=doc.get("image_content_type", "image/jpeg"),
        download_name=doc.get("image_filename", "scan.jpg"),
    )


@app.route("/result", methods=["POST"])
@login_required
def result():
    """Redirect legacy /result posts to the /scan handler."""
    # Some clients/templates may still POST directly to /result.
    # Delegate to the unified scan handler to avoid duplication.
    return scan()


@app.route("/login")
def login():
    """Render a simple login page with option to use Google SSO."""
    # If already logged in, go straight to scanner
    if "user" in session:
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/login/google")
def login_google():
    """Start Google OAuth login flow."""
    redirect_uri = url_for("auth_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@app.route("/auth/callback")
def auth_callback():
    """Handle Google OAuth callback and store user in session."""
    token = oauth.google.authorize_access_token()
    if not token:
        flash("Login failed. Please try again.")
        return redirect(url_for("login"))

    user_info = oauth.google.get("userinfo").json()
    session["user"] = {
        "id": user_info.get("id"),
        "name": user_info.get("name"),
        "email": user_info.get("email"),
        "picture": user_info.get("picture"),
    }

    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    """Log out the current user."""
    session.pop("user", None)
    return redirect(url_for("index"))


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
    app.run(host="0.0.0.0", port=5000, debug=True)
