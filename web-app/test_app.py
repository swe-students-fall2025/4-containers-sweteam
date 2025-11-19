"""Tests for the NutriBob Flask web application.

These tests exercise key routes and the temporary fake nutrition model,
ensuring authentication redirects and basic scan behavior work as expected.

Pylint configuration notes
--------------------------
- too-few-public-methods is disabled globally in this test module
    because we intentionally define tiny stub classes to stand in for
    external services like MongoDB collections and OAuth clients.
"""

# pylint: disable=too-few-public-methods

import pytest

from app import (
    MLServiceError,
    app,
    call_ml_service,
    fake_nutrition_model,
    merge_nutrition,
)


@pytest.fixture(name="client")
def fixture_client(monkeypatch, tmp_path):
    """Provide a Flask test client with a logged-in user session."""
    app.config["TESTING"] = True

    # Disable real Mongo and OAuth during tests
    monkeypatch.setattr("app.mongo_db", None, raising=False)
    monkeypatch.setattr("app.oauth", type("_O", (), {"google": None})(), raising=False)
    monkeypatch.setattr("app.ml_service_url", None, raising=False)

    # Use a temporary uploads directory
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    app.static_folder = str(tmp_path)

    test_client = app.test_client()
    with test_client.session_transaction() as sess:
        sess["user"] = {"id": "test-user", "email": "test@example.com"}
    return test_client


def test_home_redirects_to_login_when_anonymous():
    """Anonymous users visiting '/' should be redirected to the login page."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        response = test_client.get("/")
        assert response.status_code == 302
        assert "/login" in response.headers["Location"]


def test_home_redirects_to_index_when_logged_in():
    """Logged-in users visiting '/' should be redirected to the scan page."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        with test_client.session_transaction() as sess:
            sess["user"] = {"id": "u1"}
        response = test_client.get("/")
        assert response.status_code == 302
        assert "/scan" in response.headers["Location"]


def test_index_requires_login():
    """The /scan route should redirect anonymous users to /login."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        response = test_client.get("/scan")
        assert response.status_code == 302
        assert "/login" in response.headers["Location"]


def test_index_renders_for_logged_in_user(client):
    """The /scan page should render successfully for an authenticated user."""
    response = client.get("/scan")
    assert response.status_code == 200
    assert b"NutriBob" in response.data or b"Scan" in response.data


def test_scan_requires_file(client):
    """Posting to /scan without an image should redirect back to /scan."""
    response = client.post("/scan", data={})
    assert response.status_code == 302
    assert "/scan" in response.headers["Location"]


def test_scan_with_file_uses_fake_model(client, monkeypatch, tmp_path):
    """Uploading an image to /scan should invoke the fake nutrition model."""
    dummy_image_path = tmp_path / "test.jpg"
    dummy_image_path.write_bytes(b"fake image bytes")

    def fake_model(_bytes):
        return {"calories": 123, "sugar_grams": 10, "fat_grams": 2}

    monkeypatch.setattr("app.fake_nutrition_model", fake_model)

    with open(dummy_image_path, "rb") as f:
        data = {"image1": (f, "test.jpg")}
        response = client.post("/scan", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    assert b"123" in response.data


def test_fake_nutrition_model_returns_expected_keys():
    """fake_nutrition_model should return calorie, sugar, and fat keys."""
    result = fake_nutrition_model(b"test-bytes")
    assert set(result.keys()) == {"calories", "sugar_grams", "fat_grams"}


def test_login_renders_for_anonymous_user():
    """The /login page should render for anonymous users."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        response = test_client.get("/login")
        assert response.status_code == 200
        assert b"login" in response.data.lower()


def test_login_redirects_when_already_logged_in(client):
    """If already logged in, /login should redirect to /scan."""
    response = client.get("/login")
    assert response.status_code == 302
    assert "/scan" in response.headers["Location"]


def test_logout_clears_session_and_redirects():
    """/logout should clear the user session and redirect to /scan."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        with test_client.session_transaction() as sess:
            sess["user"] = {"id": "u1"}

        response = test_client.get("/logout")
        assert response.status_code == 302
        assert "/scan" in response.headers["Location"]

        with test_client.session_transaction() as sess:
            assert "user" not in sess


def test_result_route_delegates_to_scan(client, monkeypatch, tmp_path):
    """POST /result should behave like /scan (legacy compatibility)."""
    dummy_image_path = tmp_path / "legacy.jpg"
    dummy_image_path.write_bytes(b"fake image bytes")

    def fake_model(_bytes):
        return {"calories": 50, "sugar_grams": 5, "fat_grams": 1}

    monkeypatch.setattr("app.fake_nutrition_model", fake_model)

    with open(dummy_image_path, "rb") as f:
        data = {"image1": (f, "legacy.jpg")}
        response = client.post("/result", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    assert b"50" in response.data


def test_history_renders_without_db(client, monkeypatch):
    """/history should render gracefully even when the database is unavailable."""
    monkeypatch.setattr("app.mongo_db", None, raising=False)
    response = client.get("/history")
    assert response.status_code == 200


def test_history_renders_with_scans(client, monkeypatch):
    """/history should render a list of scans when DB returns results."""

    class FakeCursor:
        """Minimal iterable cursor stub for Mongo history tests."""

        def __iter__(self):
            return iter(
                [
                    {
                        "_id": "abc123",
                        "drink_name": "Milk tea",
                        "nutrition": {"calories": 100},
                    }
                ]
            )

    class FakeCollection:
        """Minimal collection stub exposing find/sort/limit methods."""

        def find(self, _query):  # noqa: D401, ARG002
            """Return a fake Mongo cursor."""
            return self

        def sort(self, _field, _direction):  # noqa: D401, ARG002
            """Return self to support chained sort()."""
            return self

        def limit(self, _n):  # noqa: D401, ARG002
            """Return a cursor limited to n items (no-op for fake)."""
            return FakeCursor()

    class FakeDB:
        """Container stub exposing a scans collection attribute."""

        scans = FakeCollection()

    monkeypatch.setattr("app.mongo_db", FakeDB(), raising=False)

    response = client.get("/history")
    assert response.status_code == 200
    assert b"Milk tea" in response.data


def test_call_ml_service_success(monkeypatch):
    """call_ml_service should return parsed JSON when the API succeeds."""

    class DummyResponse:
        """Simple stand-in for requests.Response."""

        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code
            self.text = "ok"

        def json(self):
            return self._payload

    captured = {}

    def fake_post(url, files, timeout):  # noqa: ARG001
        captured["url"] = url
        captured["files"] = files
        return DummyResponse({"success": True, "label": "milk-tea"})

    monkeypatch.setattr("app.ml_service_url", "http://ml-service")
    monkeypatch.setattr("app.requests.post", fake_post)

    result = call_ml_service(b"img", "scan.jpg", "image/jpeg")

    assert result["success"] is True
    assert captured["url"] == "http://ml-service/analyze"
    assert "image" in captured["files"]


def test_call_ml_service_raises_on_http_error(monkeypatch):
    """call_ml_service should raise MLServiceError on HTTP 400+ responses."""

    class DummyResponse:
        status_code = 500
        text = "boom"

        @staticmethod
        def json():
            return {"error": "boom"}

    def fake_post(*_args, **_kwargs):
        return DummyResponse()

    monkeypatch.setattr("app.ml_service_url", "http://ml-service")
    monkeypatch.setattr("app.requests.post", fake_post)

    with pytest.raises(MLServiceError):
        call_ml_service(b"img", "scan.jpg", "image/jpeg")


def test_call_ml_service_invalid_json(monkeypatch):
    """Invalid JSON responses should raise MLServiceError."""

    class DummyResponse:
        status_code = 200

        @staticmethod
        def json():
            raise ValueError("bad json")

    def fake_post(*_args, **_kwargs):
        return DummyResponse()

    monkeypatch.setattr("app.ml_service_url", "http://ml-service")
    monkeypatch.setattr("app.requests.post", fake_post)

    with pytest.raises(MLServiceError):
        call_ml_service(b"img", "scan.jpg", "image/jpeg")


def test_merge_nutrition_updates_summary_values():
    """merge_nutrition should override fallback values when summary is present."""
    fallback = {"calories": 100, "sugar_grams": 5, "fat_grams": 2}
    analysis_result = {
        "success": True,
        "summary": {"total_calories": 250},
        "nutrition_raw": {
            "items": [
                {"sugar_g": 10, "fat_total_g": 3.3},
                {"sugar_g": 5, "fat_total_g": 1.1},
            ]
        },
    }

    merged = merge_nutrition(fallback, analysis_result)

    assert merged["calories"] == 250
    assert merged["sugar_grams"] == 15.0
    assert merged["fat_grams"] == 4.4


def test_merge_nutrition_handles_invalid_payloads():
    """merge_nutrition should ignore malformed structures gracefully."""
    fallback = {"calories": 100, "sugar_grams": 5, "fat_grams": 2}
    analysis_result = {
        "success": True,
        "summary": "not-a-dict",
        "nutrition_raw": {"items": "not-a-list"},
    }

    merged = merge_nutrition(fallback, analysis_result)

    assert merged == fallback


def test_image_returns_503_when_db_unavailable(client, monkeypatch):
    """/image/<id> should return 503 when the database is unavailable."""
    monkeypatch.setattr("app.mongo_db", None, raising=False)
    response = client.get("/image/abc123")
    assert response.status_code == 503


def test_image_returns_404_when_scan_missing(client, monkeypatch):
    """/image/<id> should return 404 when the scan is not found."""

    class FakeScans:
        """Collection stub whose find_one always returns no document."""

        def find_one(self, _query):  # noqa: D401, ARG002
            """Return no document to simulate missing scan."""

            return None

    class FakeDB:
        """Container stub exposing scans with missing documents."""

        scans = FakeScans()

    monkeypatch.setattr("app.mongo_db", FakeDB(), raising=False)
    response = client.get("/image/abc123")
    assert response.status_code == 404


def test_auth_callback_failure_redirects_to_login(monkeypatch):
    """auth_callback should redirect to /login when token acquisition fails."""
    app.config["TESTING"] = True

    class FakeGoogleClient:
        """Stub Google client that simulates failed OAuth token retrieval."""

        def authorize_access_token(self):  # noqa: D401
            """Simulate failed token acquisition by returning None."""

            return None

    class FakeOAuth:
        """Stub OAuth container exposing a google client attribute."""

        google = FakeGoogleClient()

    monkeypatch.setattr("app.oauth", FakeOAuth(), raising=False)

    with app.test_client() as test_client:
        response = test_client.get("/auth/callback")
        assert response.status_code in {200, 302}
        assert "/login" in response.headers["Location"]
