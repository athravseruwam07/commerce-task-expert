from commerce_expert.model import ModelConfig, render_messages, render_prompt


def test_gold_never_leaks_into_prompt():
    case = {"question": "How much stock?", "schema": "CREATE TABLE inventory(qty INTEGER);", "as_of_date": "2026-01-01", "policy": "UTC", "reference_sql": "SECRET_GOLD_QUERY", "expected_rows": [{"SECRET_ANSWER": 777}], "expected_action": "query", "pattern_id": "secret_pattern"}
    prompt = render_prompt(case)
    assert "How much stock?" in prompt
    for secret in ("SECRET_GOLD_QUERY", "SECRET_ANSWER", "secret_pattern", "777"):
        assert secret not in prompt
    assert [m["role"] for m in render_messages(case)] == ["system", "user"]


def test_config_defaults():
    config = ModelConfig("models/base")
    assert config.adapter_path is None
    assert config.max_tokens == 512
    assert config.seed == 42


def test_training_preparation_never_converts_test_labels(tmp_path, monkeypatch):
    import importlib.util
    import json
    import sys
    from types import SimpleNamespace

    import pytest

    spec = importlib.util.spec_from_file_location("train_local", "scripts/train_local.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False):
            assert tokenize is False
            return json.dumps(messages)

        def encode(self, text):
            return list(range(len(text)))

    fake_auto = SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=fake_auto))
    source = tmp_path / "source"
    source.mkdir()
    case = {"id": "case", "question": "Stock?", "schema": "SCHEMA", "as_of_date": "2026-01-01", "policy": "POLICY", "expected_action": "query", "reference_sql": "SELECT 1;"}
    for split in ("train", "valid"):
        (source / f"{split}.jsonl").write_text(json.dumps(case)+"\n")
    (source / "test.jsonl").write_text("INVALID_TEST_DATA_MUST_NOT_BE_READ")
    stats = module.prepare_data(source, tmp_path / "output", tmp_path / "model", 10000)
    assert stats["train"]["count"] == 1
    assert stats["train"]["max_tokens"] > 100
    assert not (tmp_path / "output/test.jsonl").exists()
    row = json.loads((tmp_path / "output/train.jsonl").read_text())
    assert json.loads(row["messages"][-1]["content"]) == {"action": "query", "sql": "SELECT 1;"}
    with pytest.raises(ValueError, match="never silently truncate"):
        module.prepare_data(source, tmp_path / "too_long", tmp_path / "model", 10)
