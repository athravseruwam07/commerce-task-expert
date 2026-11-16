#!/usr/bin/env python3
"""Reproduce the complete local experiment, sequentially to bound memory.

Run on Apple silicon: uv run --extra ml python scripts/run_experiment.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-training", action="store_true", help="Use existing trained and quantized checkpoints"
    )
    parser.add_argument("--full", action="store_true", help="Evaluate all240 test cases into reports/full")
    args = parser.parse_args()
    run("-m", "commerce_expert.cli", "generate")
    if not args.skip_training:
        run("scripts/train_local.py", "--iters", "300", "--max-seq-length", "1024")
        run("scripts/quantize_local.py")
    report_dir = "reports/full" if args.full else "reports"
    for variant in ["base", "fine_tuned", "quantized"]:
        evaluation = ["-m", "commerce_expert.cli", "evaluate", variant, "--reports-dir", report_dir]
        if not args.full:
            evaluation += ["--limit", "60"]
        run(*evaluation)
    run("scripts/analyze_results.py", "--reports-dir", report_dir)


if __name__ == "__main__":
    main()
