import os
from functools import wraps

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
)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me-later")

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


@app.route("/history", methods=["GET"])
@login_required
def history():
    """Show a placeholder list of past scans for the logged-in user."""
    # TODO: Replace with real data from database / ML client.
    fake_scans = [
        {"drink": "Brown Sugar Boba", "calories": 380},
        {"drink": "Matcha Latte", "calories": 290},
        {"drink": "Taro Milk Tea", "calories": 340},
    ]
    return render_template(
        "history.html",
        user=session.get("user"),
        scans=fake_scans,
    )


@app.route("/result", methods=["POST"])
@login_required
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
        user=session.get("user"),
    )


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
