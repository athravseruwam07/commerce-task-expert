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
        proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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
            target = ({"action": "query", "sql": case["reference_sql"]}
                      if case["expected_action"] == "query"
                      else {"action": "clarify", "question": case.get("reference_clarification", "Which metric and time period should I use?")})
            messages = render_messages(case) + [{"role": "assistant", "content": json.dumps(target, separators=(",", ":"))}]
            tokens = tokenizer.encode(tokenizer.apply_chat_template(messages, tokenize=False))
            if len(tokens) > max_length:
                raise ValueError(f"{case['id']}: {len(tokens)} tokens exceeds {max_length}; increase sequence length, never silently truncate targets")
            lengths.append(len(tokens))
            examples.append(json.dumps({"messages": messages}))
        (output / f"{split}.jsonl").write_text("\n".join(examples) + "\n")
        stats[split] = {"count": len(examples), "max_tokens": max(lengths), "sha256": digest(source)}
    # Deliberately never read or convert test labels into training format.
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/generated")
    parser.add_argument("--model", type=Path, default=ROOT / "models/base")
    parser.add_argument("--revision", default=SOURCE_REVISION, help="Pinned Hugging Face source commit")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    raw = ROOT / "reports/raw"
    raw.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if not args.model.exists():
        from huggingface_hub import snapshot_download
        snapshot = snapshot_download(SOURCE_REPO, revision=args.revision, allow_patterns=["*.json", "*.safetensors", "*.jinja", "README.md", "LICENSE"])
        run_logged([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", snapshot, "--mlx-path", str(args.model), "--dtype", "float16"], raw / "model-convert.log")
        (args.model / "source_manifest.json").write_text(json.dumps({"repo": SOURCE_REPO, "revision": args.revision}, indent=2)+"\n")
    stats = prepare_data(args.dataset, args.dataset / "mlx", args.model, args.max_seq_length)
    metadata = {"base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "precision": "float16", "training": "LoRA (not QLoRA)", "iters": args.iters, "seed": 42, "dataset": stats, "platform": platform.platform(), "python": platform.python_version(), "versions": {p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "transformers")}}
    import yaml
    metadata["effective_config"] = yaml.safe_load((ROOT / "configs/lora.yaml").read_text())
    metadata["effective_config"].update({"iters": args.iters, "max_seq_length": args.max_seq_length, "model": str(args.model), "data": str(args.dataset / "mlx")})
    metadata["base_config_sha256"] = digest(args.model / "config.json")
    source_manifest = args.model / "source_manifest.json"
    metadata["source_revision"] = json.loads(source_manifest.read_text())["revision"] if source_manifest.exists() else "unknown (pre-existing local checkpoint)"
    metadata["status"] = "prepared" if args.prepare_only else "training"
    (ROOT / "reports/training.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not args.prepare_only:
        run_logged([sys.executable, "-m", "mlx_lm", "lora", "--config", "configs/lora.yaml", "--model", str(args.model), "--data", str(args.dataset / "mlx"), "--iters", str(args.iters), "--max-seq-length", str(args.max_seq_length)], raw / "training.log")
        run_logged([sys.executable, "-m", "mlx_lm", "fuse", "--model", str(args.model), "--adapter-path", "models/adapters", "--save-path", "models/fine_tuned"], raw / "fuse.log")
    metadata["status"] = "prepared" if args.prepare_only else "complete"
    metadata["wall_seconds"] = time.perf_counter() - start
    (ROOT / "reports/training.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
