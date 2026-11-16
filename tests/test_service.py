"""Exercise live integration without downloading or loading model weights."""
import gc
import json
import sqlite3
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest

from commerce_expert import model, service
from commerce_expert.data import AS_OF_DATE, POLICY


@pytest.fixture(autouse=True)
def clear_models():
    service.unload_model()
    yield
    service.unload_model()


@pytest.fixture
def assets(tmp_path):
    stores = tmp_path / "data" / "stores"
    stores.mkdir(parents=True)
    with sqlite3.connect(stores / "store_01.db") as connection:
        connection.execute("CREATE TABLE products(product_id INTEGER, name TEXT)")
        connection.execute("INSERT INTO products VALUES (1, 'Cup')")
    for name in ("base", "fine_tuned", "quantized"):
        (tmp_path / "models" / name).mkdir(parents=True)
    return tmp_path / "data", tmp_path / "models"


def test_live_request_uses_real_query_rows_and_policy(assets, monkeypatch):
    seen = []

    class FakeModel:
        def __init__(self, config):
            pass

        def generate(self, case):
            seen.append(case)
            return {"text": json.dumps({"action": "query", "sql": "SELECT name FROM products"}), "generation_ms": 2, "peak_memory_bytes": 100}

    monkeypatch.setattr(model, "MLXModel", FakeModel)
    result = service.answer_question("List products", "store_01", "base", *assets)
    assert result["rows"] == [{"name": "Cup"}]
    assert "Cup" in result["explanation"]
    assert seen[0]["as_of_date"] == AS_OF_DATE
    assert seen[0]["policy"] == POLICY
    assert "reference_sql" not in seen[0]
    assert result["timing"]["peak_memory_bytes"] == 100


def test_model_is_released_before_replacement(monkeypatch):
    references = []

    class FakeModel:
        def __init__(self, config):
            gc.collect()
            assert all(ref() is None for ref in references)
            references.append(weakref.ref(self))

    monkeypatch.setattr(model, "MLXModel", FakeModel)
    service._load_model("base")
    service._load_model("base")
    assert len(references) == 1
    service._load_model("quantized")
    assert len(references) == 2 and references[0]() is None


def test_clarification_and_failure_remain_visible(assets, monkeypatch):
    class FakeModel:
        def __init__(self, config):
            pass

        def generate(self, case):
            if "ambiguous" in case["question"]:
                return {"text": '{"action":"clarify","question":"Which ranking metric?"}'}
            raise RuntimeError("inference failed")

    monkeypatch.setattr(model, "MLXModel", FakeModel)
    result = service.answer_question("ambiguous", "store_01", "base", *assets)
    assert result["clarification"] == "Which ranking metric?" and not result["rows"]
    failed = service.answer_question("failure", "store_01", "base", *assets)
    assert failed["error"] == "RuntimeError: inference failed"


def test_unknown_store_and_path_escape_are_rejected(assets):
    assert service.answer_question("list", "missing", "base", *assets)["error"] == "Unknown store"
    assert service.answer_question("list", "../../outside", "base", *assets)["error"] == "Unknown store"


def test_concurrent_requests_share_only_one_loaded_model(assets, monkeypatch):
    loads = []

    class FakeModel:
        def __init__(self, config):
            loads.append(config.model_path)

        def generate(self, case):
            return {"text": '{"action":"query","sql":"SELECT COUNT(*) AS n FROM products"}'}

    monkeypatch.setattr(model, "MLXModel", FakeModel)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: service.answer_question("count", "store_01", "base", *assets), range(3)))
    assert len(loads) == 1
    assert all(result["rows"] == [{"n": 1}] for result in results)
