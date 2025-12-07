# tests/test_api.py

from pathlib import Path
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_score_email_mock_chase():
    """
    Basic smoke test: send the mock_chase_alert.eml through the API
    and verify we get a label and score fields.
    """
    # Adjust path if your mock file lives somewhere else
    eml_path = Path("mock_chase_alert.eml")
    if not eml_path.is_file():
        # If you moved it under data/raw, adjust here:
        alt = Path("data/raw/mock_chase_alert.eml")
        if alt.is_file():
            eml_path = alt
        else:
            # Skip test if file missing
            return

    with eml_path.open("rb") as f:
        files = {"file": ("mock_chase_alert.eml", f, "message/rfc822")}
        resp = client.post("/score-email", files=files)

    assert resp.status_code == 200
    body = resp.json()

    # Basic keys
    assert "label" in body
    assert "phish_score" in body
    assert "text_score" in body

    # Sanity checks
    assert body["phish_score"] >= 0.0
    assert body["phish_score"] <= 1.0
