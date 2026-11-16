#!/usr/bin/env python3
"""Quantize the fused fine-tuned checkpoint, preserving the runtime and tokenizer."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "models/fine_tuned")
    parser.add_argument("--output", type=Path, default=ROOT / "models/quantized")
    args = parser.parse_args()
    if not (args.source / "config.json").exists():
        raise SystemExit("Train and fuse the fine-tuned model before quantizing it.")
    start = time.perf_counter()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_lm",
            "convert",
            "--hf-path",
            str(args.source),
            "--mlx-path",
            str(args.output),
            "-q",
            "--q-bits",
            "4",
            "--q-group-size",
            "64",
            "--dtype",
            "float16",
        ],
        check=True,
        cwd=ROOT,
    )
    metadata = {
        "source": str(args.source.relative_to(ROOT) if args.source.is_relative_to(ROOT) else args.source),
        "output": str(args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else args.output),
        "bits": 4,
        "group_size": 64,
        "method": "MLX affine weight-only quantization",
        "wall_seconds": time.perf_counter() - start,
        "artifacts": {
            str(p.relative_to(args.output)): p.stat().st_size for p in args.output.rglob("*") if p.is_file()
        },
    }
    metadata["source_weights_sha256"] = {p.name: file_sha256(p) for p in args.source.glob("*.safetensors")}
    metadata["quantized_weights_sha256"] = {p.name: file_sha256(p) for p in args.output.glob("*.safetensors")}
    (ROOT / "reports/quantization.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
