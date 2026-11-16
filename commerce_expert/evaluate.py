"""Paired, first-attempt evaluation. No test examples enter training or prompts."""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import time
from collections import defaultdict
from pathlib import Path

from commerce_expert.query import execute_readonly, parse_prediction, rows_equal


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def explain(rows: list[dict], policy: str = "") -> str:
    """Deterministic evidence rendering keeps prose grounded in returned rows."""
    if not rows:
        return "No matching results were returned under the stated assumptions."
    parts = []
    for row in rows[:5]:
        parts.append("; ".join(f"{key.replace('_', ' ')}: {value}" for key, value in row.items()))
    suffix = f" Showing 5 of {len(rows)} rows." if len(rows) > 5 else ""
    return "\n".join(parts) + suffix


def evaluate_prediction(case: dict, prediction: dict, data_dir: Path) -> dict:
    started = time.perf_counter()
    result = dict(prediction)
    if prediction.get("error"):
        result["generation_error"] = prediction["error"]
    result.update(
        action=None, sql=None, rows=[], error=None, clarification=None, correct=False, execution_success=False
    )
    try:
        parsed = parse_prediction(prediction["text"])
        result["action"] = parsed["action"]
        if parsed["action"] == "clarify":
            result["clarification"] = parsed["question"]
            result["correct"] = case["expected_action"] == "clarify" and bool(parsed["question"].strip())
        else:
            result["sql"] = parsed["sql"]
            db = (Path(data_dir) / case["db_path"]).resolve()
            if not db.is_relative_to(Path(data_dir).resolve()):
                raise ValueError("Database path escapes dataset directory")
            result["rows"] = execute_readonly(db, parsed["sql"])
            result["execution_success"] = True
            result["correct"] = case["expected_action"] == "query" and rows_equal(
                result["rows"], case["expected_rows"], case.get("ordered", False)
            )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["explanation"] = (
        explain(result["rows"], case.get("policy", "")) if result["execution_success"] else ""
    )
    result["scoring_ms"] = (time.perf_counter() - started) * 1000
    return result


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * q))]


def summarize(records: list[dict], variant: str) -> dict:
    queries = [r for r in records if r["expected_action"] == "query"]
    ambiguous = [r for r in records if r["expected_action"] == "clarify"]
    slices = defaultdict(list)
    for r in records:
        slices[r["slice"]].append(r)
    times = [r["generation_ms"] for r in records if r.get("generation_ms") is not None]
    return {
        "variant": variant,
        "n": len(records),
        "answer_accuracy": sum(bool(r["correct"]) for r in records) / len(records) if records else None,
        "query_answer_accuracy": sum(bool(r["correct"]) for r in queries) / len(queries) if queries else None,
        "query_execution_rate": sum(bool(r["execution_success"]) for r in queries) / len(queries)
        if queries
        else None,
        "clarification_accuracy": sum(bool(r["correct"]) for r in ambiguous) / len(ambiguous)
        if ambiguous
        else None,
        "p50_generation_ms": statistics.median(times) if times else None,
        "p95_generation_ms": percentile(times, 0.95),
        "peak_memory_bytes": max((r.get("peak_memory_bytes", 0) or 0 for r in records), default=0),
        "slices": {
            s: {"n": len(rs), "accuracy": sum(bool(r["correct"]) for r in rs) / len(rs)}
            for s, rs in slices.items()
        },
    }


def select_cases(cases: list[dict], limit: int | None = None) -> list[dict]:
    """Round robin across slices and patterns for a deterministic pilot subset."""
    if not limit or limit >= len(cases):
        return cases
    groups = defaultdict(lambda: defaultdict(list))
    for case in cases:
        groups[case["slice"]][case["pattern_id"]].append(case)
    selected = []
    offsets = defaultdict(int)
    while len(selected) < limit:
        made_progress = False
        for name in sorted(groups):
            patterns = sorted(groups[name])
            for _ in patterns:
                key = patterns[offsets[name] % len(patterns)]
                offsets[name] += 1
                if groups[name][key]:
                    selected.append(groups[name][key].pop(0))
                    made_progress = True
                    break
            if len(selected) == limit:
                break
        if not made_progress:
            break
    return selected


def run_evaluation(
    data_dir: Path,
    model_path: Path,
    variant: str,
    reports_dir: Path,
    limit: int | None = None,
    max_tokens: int = 512,
) -> dict:
    from commerce_expert.model import MLXModel, ModelConfig

    test_path = Path(data_dir) / "test.jsonl"
    cases = select_cases(read_jsonl(test_path), limit)
    model = MLXModel(ModelConfig(model_path=str(model_path), max_tokens=max_tokens))
    # Warm up on validation only, so timing excludes model load and first compilation.
    valid = read_jsonl(Path(data_dir) / "valid.jsonl")
    if valid:
        model.generate(valid[0])
    output = Path(reports_dir) / "records" / f"{variant}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with output.open("w") as f:
        for index, case in enumerate(cases):
            started = time.perf_counter()
            try:
                generated = model.generate(case)
            except Exception as exc:
                generated = {
                    "text": "",
                    "generation_ms": (time.perf_counter() - started) * 1000,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            scored = evaluate_prediction(case, generated, data_dir)
            record = {
                key: case.get(key)
                for key in [
                    "question",
                    "store_id",
                    "slice",
                    "pattern_id",
                    "expected_action",
                    "expected_rows",
                    "reference_sql",
                    "policy",
                    "as_of_date",
                ]
            }
            record.update(
                scored, case_id=case["id"], variant=variant, total_ms=(time.perf_counter() - started) * 1000
            )
            records.append(record)
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(f"{variant} {index + 1}/{len(cases)} correct={record['correct']}", flush=True)
    summary = summarize(records, variant)
    summary["test_sha256"] = hashlib.sha256(test_path.read_bytes()).hexdigest()
    summary["case_ids"] = [case["id"] for case in cases]
    summary["model_path"] = str(model_path)
    summary["max_tokens"] = max_tokens
    path = Path(reports_dir) / f"{variant}.json"
    path.write_text(json.dumps(summary, indent=2))
    merge_reports(reports_dir)
    return summary


def merge_reports(reports_dir: Path) -> dict:
    runs = []
    for variant in ["base", "fine_tuned", "quantized"]:
        path = Path(reports_dir) / f"{variant}.json"
        if path.exists():
            runs.append(json.loads(path.read_text()))
    value = {
        "runs": runs,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "chip": "Apple M4",
            "unified_memory_gb": 16,
        },
        "protocol": {
            "decoding": "greedy, first attempt, no repair",
            "batch_size": 1,
            "warmup": "one validation example per model",
            "runtime": "MLX LM",
            "answer_rendering": "deterministic row-to-text renderer; not a learned prose model",
            "memory": "peak MLX allocation during generation; unified memory, not CUDA VRAM",
            "clarification_metric": "decision correctness only; clarification wording requires manual audit",
        },
    }
    (Path(reports_dir) / "summary.json").write_text(json.dumps(value, indent=2))
    return value
