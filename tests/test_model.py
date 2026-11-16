from commerce_expert.model import ModelConfig, render_messages, render_prompt


def test_gold_never_leaks_into_prompt():
    case = {"question": "How much stock?", "schema": "CREATE TABLE inventory(qty INTEGER);", "as_of_date": "2026-01-01", "policy": "UTC", "reference_sql": "SECRET_GOLD_QUERY", "expected_rows": [{"SECRET_ANSWER": 777}], "expected_action": "query", "pattern_id": "secret_pattern", "output_columns": ["SECRET_OUTPUT_COLUMN"]}
    prompt = render_prompt(case)
    assert "How much stock?" in prompt
    for secret in ("SECRET_GOLD_QUERY", "SECRET_ANSWER", "secret_pattern", "SECRET_OUTPUT_COLUMN", "777"):
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


def test_training_curve_parses_rounded_console_records(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_curve", "scripts/train_local.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    log = tmp_path / "training.log"
    log.write_text("Trainable parameters: 0.085%\n 1 val \x1b[1m0.798\x1b[0m 6.88s\n 10 \x1b[1m0.578 ▼\x1b[0m 64 0.9k\n")
    assert module.extract_curve(log) == [{"step": 1, "validation_loss": 0.798, "validation_seconds": 6.88}, {"step": 10, "train_loss": 0.578}]


def test_checkpoint_selection_uses_only_validation_and_picks_earliest_loss_tie(tmp_path, monkeypatch):
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location("train_selection", "scripts/train_local.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    dataset = tmp_path / "dataset"
    (dataset / "mlx").mkdir(parents=True)
    (dataset / "valid.jsonl").write_text('\n'.join(json.dumps({"id": f"valid{i}", "pattern_id": f"pattern{i}"}) for i in range(2))+"\n")
    (dataset / "test.jsonl").write_text("INVALID_TEST_MUST_NOT_BE_READ")
    (dataset / "mlx/valid.jsonl").write_text('{"messages": ["validation0"]}\n{"messages": ["validation1"]}\n')
    adapters = tmp_path / "models/adapters"
    adapters.mkdir(parents=True)
    (tmp_path / "reports/raw").mkdir(parents=True)
    (adapters / "adapter_config.json").write_text("{}")
    (adapters / "adapters.safetensors").write_text("final300")
    for step in (100, 200, 300):
        (adapters / f"{step:07d}_adapters.safetensors").write_text(f"weights{step}")

    def fake_run(command, logfile):
        assert "--mask-prompt" in command
        step = int(logfile.name.split("_")[1])
        logfile.write_text(f"Test loss {0.2 if step == 100 else 0.1:.3f}, Test ppl 1.000.\n")

    monkeypatch.setattr(module, "run_logged", fake_run)
    selection = module.select_checkpoint(dataset, tmp_path / "base", 300, 1024)
    assert selection["test_used"] is False
    assert selection["selected_iteration"] == 200
    assert set(selection["validation_case_ids"]) == {"valid0", "valid1"}
    assert (adapters / "adapters.safetensors").read_text() == "weights200"
    assert "validation0" in (dataset / "validation_selection/test.jsonl").read_text()
