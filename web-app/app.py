"""Flask web application for NutriBob milk tea nutrition scanning.

This module sets up the web UI, Google SSO authentication, and routes for
uploading images to be analyzed by the (currently fake) nutrition model.
"""

import os
import io
from datetime import datetime
from functools import wraps
from typing import Any
import certifi
import requests

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
from bson import ObjectId

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me-later")

mongo_client: MongoClient | None = None
mongo_db = None

mongo_uri = os.environ.get("MONGODB_URI")
ml_service_url = os.environ.get("ML_SERVICE_URL")
if ml_service_url:
    ml_service_url = ml_service_url.rstrip("/")

MAX_IMAGE_SIZE_BYTES = 16 * 1024 * 1024  # 16 MB upload limit

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


class MLServiceError(Exception):
    """Raised when the external ML analysis service cannot be used."""


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
    image_file = image_file1
    if image_file1 is None or image_file1.filename == "":
        image_file = image_file2

    if image_file is None or image_file.filename == "":
        flash("Please select or scan a milk tea image first.")
        return redirect(url_for("index"))

    image_bytes = image_file.read()
    if not image_bytes:
        flash("Uploaded image is empty. Please try again.")
        return redirect(url_for("index"))

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        flash("Image too large. Please upload an image under 16MB.")
        return redirect(url_for("index"))

    image_filename = image_file.filename or "scan.jpg"
    image_content_type = image_file.mimetype or "application/octet-stream"

    nutrition = fake_nutrition_model(image_bytes)
    analysis_result: dict[str, Any] | None = None
    if ml_service_url:
        try:
            analysis_result = call_ml_service(
                image_bytes, image_filename, image_content_type
            )
            nutrition = merge_nutrition(nutrition, analysis_result)
        except MLServiceError as exc:
            print(f"ML service request failed: {exc}")
            flash(
                "Milk tea model is unavailable right now; showing sample values.",
                "warning",
            )

    user = session.get("user")
    scan_doc_id = None
    if mongo_db is not None and user:
        try:
            scan_doc = {
                "user_id": user.get("id"),
                "user_email": user.get("email"),
                "image_filename": image_filename,
                "image_content_type": image_content_type,
                "image_data": Binary(image_bytes),
                "drink_name": (
                    (analysis_result or {}).get("label") if analysis_result else None
                )
                or "Milk tea",
                "nutrition": nutrition,
                "created_at": datetime.utcnow(),
            }
            if analysis_result is not None:
                scan_doc["analysis_result"] = analysis_result

            insert_result = mongo_db.scans.insert_one(scan_doc)
            scan_doc_id = str(insert_result.inserted_id)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Mongo insert failed: {exc}")

    return render_template(
        "result.html",
        nutrition=nutrition,
        scan_id=scan_doc_id,
        ml_result=analysis_result,
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


def call_ml_service(
    image_bytes: bytes, filename: str, content_type: str
) -> dict[str, Any]:
    """Send an image to the ML service and return its JSON payload."""
    if not ml_service_url:
        raise MLServiceError("ML_SERVICE_URL is not configured.")

    endpoint = f"{ml_service_url}/analyze"
    files = {"image": (filename, image_bytes, content_type)}
    try:
        response = requests.post(endpoint, files=files, timeout=30)
    except requests.RequestException as exc:
        raise MLServiceError(f"Request to ML service failed: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise MLServiceError("ML service returned invalid JSON.") from exc

    if response.status_code >= 400:
        error_message = (
            payload.get("error") if isinstance(payload, dict) else response.text
        )
        raise MLServiceError(f"ML service error: {error_message}")
    return payload


def merge_nutrition(
    fallback: dict[str, Any], analysis_result: dict[str, Any] | None
) -> dict[str, Any]:
    """Combine fallback nutrition values with analysis results when available."""
    merged = dict(fallback)
    if not analysis_result or not analysis_result.get("success"):
        return merged

    summary = analysis_result.get("summary") or {}
    if not isinstance(summary, dict):
        summary = {}
    calories = summary.get("total_calories")
    if calories is not None:
        merged["calories"] = round(calories)

    raw = analysis_result.get("nutrition_raw") or {}
    if not isinstance(raw, dict):
        raw = {}
    items = raw.get("items") or []
    try:
        iterator = [item for item in list(items) if isinstance(item, dict)]
    except TypeError:
        iterator = []

    if iterator:
        sugar_total = sum(item.get("sugar_g", 0) for item in iterator)
        fat_total = sum(item.get("fat_total_g", 0) for item in iterator)
        if sugar_total is not None:
            merged["sugar_grams"] = round(sugar_total, 1)
        if fat_total is not None:
            merged["fat_grams"] = round(fat_total, 1)

    return merged


def fake_nutrition_model(image_bytes: bytes) -> dict:
    """
    Temporary fake model.

    Args:
        image_bytes: Raw bytes of the uploaded drink image (currently unused).

    Returns:
        A dictionary with dummy nutrition information.
    """
    # Mark the argument as intentionally unused for now
    _ = image_bytes

    return {
        "calories": 380,
        "sugar_grams": 38,
        "fat_grams": 8,
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
