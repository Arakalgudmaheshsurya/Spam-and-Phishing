from pathlib import Path
from src.inference.pipeline import extract_all


def test_pipeline_on_mock_eml():
    eml_path = Path("mock_chase_alert.eml")
    if not eml_path.is_file():
        return  # skip if file not there

    raw = eml_path.read_bytes()
    result = extract_all(raw)

    assert "label" in result
    assert "phish_score" in result
    assert result["phish_score"] >= 0.0
    assert result["phish_score"] <= 1.0
