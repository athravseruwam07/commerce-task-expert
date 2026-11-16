import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "analyze", Path(__file__).resolve().parents[1] / "scripts/analyze_results.py"
)
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)


def test_paired_requires_same_cases():
    with pytest.raises(ValueError):
        analyze.paired_bootstrap([{"case_id": "a", "correct": True}], [{"case_id": "b", "correct": True}])


def test_paired_bootstrap_reproducible():
    left = [{"case_id": str(i), "correct": False} for i in range(5)]
    right = [{"case_id": str(i), "correct": True} for i in range(5)]
    result = analyze.paired_bootstrap(left, right, 100)
    assert result["delta"] == 1 and result["ci95"] == [1, 1]
