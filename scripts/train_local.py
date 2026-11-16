#!/usr/bin/env python3
"""Prepare verified chat data, train local LoRA, and fuse its FP16 checkpoint.

Usage: uv run --extra ml python scripts/train_local.py [--iters 300]
All models and computations stay local; only public model weights are downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SOURCE_REPO = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
SOURCE_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from commerce_expert.model import render_messages  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_logged(command: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as logfile:
        proc = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                logfile.write(line)
                logfile.flush()
        except BaseException:
            proc.terminate()
            proc.wait(timeout=30)
            raise
        if proc.wait() != 0:
            raise RuntimeError(f"Command failed; see {path}")


def prepare_data(dataset: Path, output: Path, model: Path, max_length: int) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    output.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split in ("train", "valid"):
        source = dataset / f"{split}.jsonl"
        examples = []
        lengths = []
        for line in source.read_text().splitlines():
            case = json.loads(line)
            target = (
                {"action": "query", "sql": case["reference_sql"]}
                if case["expected_action"] == "query"
                else {
                    "action": "clarify",
                    "question": case.get(
                        "reference_clarification", "Which metric and time period should I use?"
                    ),
                }
            )
            messages = render_messages(case) + [
                {"role": "assistant", "content": json.dumps(target, separators=(",", ":"))}
            ]
            tokens = tokenizer.encode(tokenizer.apply_chat_template(messages, tokenize=False))
            if len(tokens) > max_length:
                raise ValueError(
                    f"{case['id']}: {len(tokens)} tokens exceeds {max_length}; increase sequence length, never silently truncate targets"
                )
            lengths.append(len(tokens))
            examples.append(json.dumps({"messages": messages}))
        (output / f"{split}.jsonl").write_text("\n".join(examples) + "\n")
        stats[split] = {"count": len(examples), "max_tokens": max(lengths), "sha256": digest(source)}
    # Deliberately never read or convert test labels into training format.
    return stats


def extract_curve(log_path: Path) -> list[dict]:
    """Read rounded console losses as an auditable compact training trace."""
    clean = re.sub(r"\x1b\[[0-9;]*m", "", log_path.read_text())
    result = []
    for line in clean.splitlines():
        val = re.match(r"^\s*(\d+)\s+val\s+([0-9.]+)\s+([0-9.]+)s", line)
        train = re.match(r"^\s*(\d+)\s+([0-9.]+)\s", line)
        if val:
            result.append(
                {"step": int(val[1]), "validation_loss": float(val[2]), "validation_seconds": float(val[3])}
            )
        elif train:
            result.append({"step": int(train[1]), "train_loss": float(train[2])})
    return result


def select_checkpoint(dataset: Path, model: Path, iters: int, max_length: int) -> dict:
    """Choose checkpoint by completion loss on identical fixed validation cases."""
    from collections import defaultdict

    groups = defaultdict(list)
    cases = [json.loads(line) for line in (dataset / "valid.jsonl").read_text().splitlines()]
    converted = (dataset / "mlx/valid.jsonl").read_text().splitlines()
    for case, row in zip(cases, converted, strict=True):
        groups[case["pattern_id"]].append((case["id"], row))
    for rows in groups.values():
        rows.sort(key=lambda item: hashlib.sha256(item[0].encode()).hexdigest())
    selected = []
    while len(selected) < min(40, len(cases)):
        for key in sorted(groups):
            if groups[key] and len(selected) < min(40, len(cases)):
                selected.append(groups[key].pop(0))
    selection_data = dataset / "validation_selection"
    selection_data.mkdir(exist_ok=True)
    # MLX's loss-only CLI names this test.jsonl. These are VALIDATION cases,
    # not a single case or answer from the benchmark test split.
    (selection_data / "test.jsonl").write_text("\n".join(row for _, row in selected) + "\n")
    candidates = sorted(set(range(100, iters + 1, 100)) | {iters})
    results = []
    adapters = ROOT / "models/adapters"
    for step in candidates:
        candidate = ROOT / f"models/checkpoint_selection/step_{step}"
        candidate.mkdir(parents=True, exist_ok=True)
        shutil.copy(adapters / "adapter_config.json", candidate / "adapter_config.json")
        weights = adapters / f"{step:07d}_adapters.safetensors"
        if not weights.exists() and step == iters:
            weights = adapters / "adapters.safetensors"
        if not weights.exists():
            raise FileNotFoundError(f"Missing checkpoint {step}: {weights}")
        shutil.copy(weights, candidate / "adapters.safetensors")
        logfile = ROOT / f"reports/raw/checkpoint_{step}_validation.log"
        run_logged(
            [
                sys.executable,
                "-m",
                "mlx_lm",
                "lora",
                "--test",
                "--model",
                str(model),
                "--adapter-path",
                str(candidate),
                "--data",
                str(selection_data),
                "--test-batches",
                "-1",
                "--batch-size",
                "1",
                "--mask-prompt",
                "--max-seq-length",
                str(max_length),
                "--seed",
                "42",
            ],
            logfile,
        )
        clean = re.sub(r"\x1b\[[0-9;]*m", "", logfile.read_text())
        match = re.search(r"Test loss ([0-9.]+),", clean)
        if not match:
            raise RuntimeError(f"Unable to parse validation loss in {logfile}")
        results.append({"step": step, "validation_completion_loss": float(match[1])})
    winner = min(results, key=lambda row: (row["validation_completion_loss"], row["step"]))
    final = adapters / f"{iters:07d}_adapters.safetensors"
    if not final.exists():
        shutil.copy(adapters / "adapters.safetensors", final)
    shutil.copy(
        ROOT / f"models/checkpoint_selection/step_{winner['step']}/adapters.safetensors",
        adapters / "adapters.safetensors",
    )
    selection = {
        "criterion": "Minimum completion-only cross-entropy on the same fixed, pattern-stratified 40 validation cases; console losses rounded to 3 decimals; ties choose earliest checkpoint",
        "selected_iteration": winner["step"],
        "validation_case_ids": [case_id for case_id, _ in selected],
        "candidates": results,
        "test_used": False,
    }
    (ROOT / "reports/checkpoint_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/generated")
    parser.add_argument("--model", type=Path, default=ROOT / "models/base")
    parser.add_argument("--revision", default=SOURCE_REVISION, help="Pinned Hugging Face source commit")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--select-only", action="store_true", help="Select/fuse existing checkpoints without retraining"
    )
    args = parser.parse_args()
    raw = ROOT / "reports/raw"
    raw.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if not args.model.exists():
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            SOURCE_REPO,
            revision=args.revision,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "README.md", "LICENSE"],
        )
        run_logged(
            [
                sys.executable,
                "-m",
                "mlx_lm",
                "convert",
                "--hf-path",
                snapshot,
                "--mlx-path",
                str(args.model),
                "--dtype",
                "float16",
            ],
            raw / "model-convert.log",
        )
        (args.model / "source_manifest.json").write_text(
            json.dumps({"repo": SOURCE_REPO, "revision": args.revision}, indent=2) + "\n"
        )
    stats = prepare_data(args.dataset, args.dataset / "mlx", args.model, args.max_seq_length)
    metadata = {
        "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "precision": "float16",
        "training": "LoRA (not QLoRA)",
        "iters": args.iters,
        "seed": 42,
        "dataset": stats,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "transformers")},
    }
    import yaml

    metadata["effective_config"] = yaml.safe_load((ROOT / "configs/lora.yaml").read_text())
    metadata["effective_config"].update(
        {
            "iters": args.iters,
            "max_seq_length": args.max_seq_length,
            "model": str(args.model),
            "data": str(args.dataset / "mlx"),
        }
    )
    metadata["base_config_sha256"] = digest(args.model / "config.json")
    source_manifest = args.model / "source_manifest.json"
    metadata["source_revision"] = (
        json.loads(source_manifest.read_text())["revision"]
        if source_manifest.exists()
        else "unknown (pre-existing local checkpoint)"
    )
    record_path = ROOT / ("reports/preparation.json" if args.prepare_only else "reports/training.json")
    if args.select_only and record_path.exists():
        previous = json.loads(record_path.read_text())
        if "training_seconds" in previous:
            metadata["training_seconds"] = previous["training_seconds"]
    metadata["status"] = "prepared" if args.prepare_only else "training"
    record_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if not args.prepare_only:
        if not args.select_only:
            training_start = time.perf_counter()
            run_logged(
                [
                    sys.executable,
                    "-m",
                    "mlx_lm",
                    "lora",
                    "--config",
                    "configs/lora.yaml",
                    "--model",
                    str(args.model),
                    "--data",
                    str(args.dataset / "mlx"),
                    "--iters",
                    str(args.iters),
                    "--max-seq-length",
                    str(args.max_seq_length),
                ],
                raw / "training.log",
            )
            metadata["training_seconds"] = time.perf_counter() - training_start
        metadata["checkpoint_selection"] = select_checkpoint(
            args.dataset, args.model, args.iters, args.max_seq_length
        )
        run_logged(
            [
                sys.executable,
                "-m",
                "mlx_lm",
                "fuse",
                "--model",
                str(args.model),
                "--adapter-path",
                "models/adapters",
                "--save-path",
                "models/fine_tuned",
            ],
            raw / "fuse.log",
        )
    if not args.prepare_only:
        curve = {
            "source": "MLX console output (losses rounded to three decimals)",
            "events": extract_curve(raw / "training.log"),
        }
        (ROOT / "reports/training_curve.json").write_text(json.dumps(curve, indent=2) + "\n")
    metadata["status"] = "prepared" if args.prepare_only else "complete"
    metadata["wall_seconds"] = time.perf_counter() - start
    record_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
