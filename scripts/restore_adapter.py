#!/usr/bin/env python3
"""Restore the released tiny adapter, fuse it locally, and make the 4-bit model."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

from train_local import ROOT, SOURCE_REPO, SOURCE_REVISION


def restore_archive(archive: Path, destination: Path) -> None:
    """Extract exactly two bounded regular files after verifying their recorded hashes."""
    with tarfile.open(archive, "r:gz") as tar:
        allowed = {"adapter_config.json", "adapters.safetensors", "manifest.json"}
        members = tar.getmembers()
        if {m.name for m in members} != allowed or len(members) != 3:
            raise ValueError("Unexpected release archive members")
        payloads = {}
        for member in members:
            if not member.isfile() or member.size > 16 * 1024 * 1024:
                raise ValueError("Archive member is not a bounded regular file")
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError("Missing archive payload")
            payloads[member.name] = stream.read()
    manifest = json.loads(payloads["manifest.json"])
    if manifest["source_repo"] != SOURCE_REPO or manifest["source_revision"] != SOURCE_REVISION:
        raise ValueError("Adapter base revision does not match the pinned project model")
    for name in allowed - {"manifest.json"}:
        if hashlib.sha256(payloads[name]).hexdigest() != manifest["sha256"][name]:
            raise ValueError(f"Adapter checksum mismatch: {name}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (destination / name).write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    restore_archive(args.archive, ROOT / "models/adapters")
    base = ROOT / "models/base"
    if not base.exists():
        from huggingface_hub import snapshot_download

        source = snapshot_download(
            SOURCE_REPO,
            revision=SOURCE_REVISION,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "README.md", "LICENSE"],
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mlx_lm",
                "convert",
                "--hf-path",
                source,
                "--mlx-path",
                str(base),
                "--dtype",
                "float16",
            ],
            cwd=ROOT,
            check=True,
        )
        (base / "source_manifest.json").write_text(
            json.dumps({"repo": SOURCE_REPO, "revision": SOURCE_REVISION})
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_lm",
            "fuse",
            "--model",
            str(base),
            "--adapter-path",
            "models/adapters",
            "--save-path",
            "models/fine_tuned",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run([sys.executable, "scripts/quantize_local.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
