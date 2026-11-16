"""Test evidence-loading boundaries; model inference is covered separately."""
import json
from pathlib import Path

from app import available_stores, read_records, summary_frame


def test_store_manifest_and_file_fallback(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"stores": [{"store_id": "north"}, {"id": "south"}]}))
    assert available_stores(tmp_path) == ["north", "south"]
    (tmp_path / "manifest.json").write_text("broken")
    (tmp_path / "stores").mkdir()
    (tmp_path / "stores" / "fallback.db").touch()
    assert available_stores(tmp_path) == ["fallback"]


def test_read_records_does_not_invent_missing_runs(tmp_path):
    assert read_records(tmp_path) == []
    (tmp_path / "base.jsonl").write_text(json.dumps({"question": "How many orders?", "error": "invalid SQL"}) + "\n")
    records = read_records(tmp_path)
    assert records[0]["error"] == "invalid SQL"
    assert records[0]["_record_file"] == "base.jsonl"


def test_summary_retains_actual_values_only():
    assert summary_frame(None).empty
    frame = summary_frame({"variants": {"base": {"accuracy": 0.4, "count": 20}}})
    assert frame.iloc[0]["accuracy"] == 0.4
    assert "latency_ms" not in frame


def test_app_renders_without_generated_artifacts():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=30)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Ask your store", "Experiment results", "How it works"]
    assert any("Know your store" in title.value for title in app.title)


def test_summary_supports_evaluation_protocol():
    frame = summary_frame({"runs": [{"variant": "base", "n": 12, "answer_accuracy": .25, "slices": {"unseen_store": {}}}], "hardware": {"memory": 16}})
    assert frame.iloc[0]["variant"] == "base"
    assert "slices" not in frame.columns
