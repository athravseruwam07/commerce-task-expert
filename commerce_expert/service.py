"""Shared local inference service for CLI and demo.

A process owns at most one resident model. The lock spans loading and generation:
Streamlit sessions cannot interleave cache eviction with another request's model.
"""
from __future__ import annotations

import gc
import threading
import time
from pathlib import Path
from typing import Any

from commerce_expert.evaluate import explain
from commerce_expert.query import execute_readonly, parse_prediction

_MODEL: Any = None
_MODEL_PATH: str | None = None
_MODEL_LOCK = threading.RLock()


def unload_model() -> None:
    """Release model weights before loading a replacement, including Metal caches."""
    global _MODEL, _MODEL_PATH
    with _MODEL_LOCK:
        mx = getattr(_MODEL, "mx", None)
        if mx is not None:
            mx.synchronize()
        _MODEL = None
        _MODEL_PATH = None
        gc.collect()
        if mx is not None:
            mx.clear_cache()


def _load_model(model_path: str):
    global _MODEL, _MODEL_PATH
    with _MODEL_LOCK:
        if _MODEL is not None and _MODEL_PATH == model_path:
            return _MODEL
        unload_model()
        from commerce_expert.model import MLXModel, ModelConfig

        _MODEL = MLXModel(ModelConfig(model_path=model_path))
        _MODEL_PATH = model_path
        return _MODEL


def answer_question(
    question: str,
    store_id: str,
    variant: str,
    data_dir: Path = Path("data/generated"),
    models_dir: Path = Path("models"),
) -> dict:
    from commerce_expert.data import AS_OF_DATE, POLICY, SCHEMA

    data_dir = Path(data_dir).resolve()
    stores_dir = (data_dir / "stores").resolve()
    path = (stores_dir / f"{store_id}.db").resolve()
    if not path.is_relative_to(stores_dir) or not path.is_file():
        return {"error": "Unknown store", "action": None, "rows": []}
    if variant not in {"base", "fine_tuned", "quantized"}:
        return {"error": "Unknown model variant", "action": None, "rows": []}
    model_path = (Path(models_dir) / variant).resolve()
    if not model_path.is_dir():
        return {
            "error": f"Model unavailable: {variant}. Run the local training pipeline first.",
            "action": None,
            "rows": [],
        }
    case = {"question": question, "schema": SCHEMA, "policy": POLICY, "as_of_date": AS_OF_DATE}
    started = time.perf_counter()
    result = {
        "action": None, "sql": None, "rows": [], "explanation": "", "error": None,
        "clarification": None, "policy": case["policy"],
        "assumptions": {"as_of_date": AS_OF_DATE, "business_policy": POLICY},
    }
    try:
        with _MODEL_LOCK:
            generated = _load_model(str(model_path)).generate(case)
        result.update(generated)
        parsed = parse_prediction(generated["text"])
        result["action"] = parsed["action"]
        if parsed["action"] == "query":
            result["sql"] = parsed["sql"]
            result["rows"] = execute_readonly(path, parsed["sql"])
            result["explanation"] = explain(result["rows"], case["policy"])
        else:
            result["clarification"] = parsed["question"]
    except Exception as exc:  # noqa: BLE001 -- request boundary preserves actual inference/execution failures.
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["timing"] = {
        "total_ms": (time.perf_counter() - started) * 1000,
        **{key: result[key] for key in ("generation_ms", "first_token_ms", "peak_memory_bytes", "prompt_tokens", "completion_tokens") if key in result},
    }
    return result
