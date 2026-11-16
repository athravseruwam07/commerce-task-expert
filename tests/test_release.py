import importlib.util
import io
import sys
import tarfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("restore", SCRIPTS / "restore_adapter.py")
restore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore)


def test_restore_rejects_unexpected_path(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../escaped")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        restore.restore_archive(archive, tmp_path / "dest")
    assert not (tmp_path / "escaped").exists()
