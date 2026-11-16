import copy
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from commerce_expert.benchmark import (
    BASE_PATTERNS,
    HELDOUT_PATTERNS,
    _equal,
    build_sql,
    generate_dataset,
    oracle,
)
from commerce_expert.data import create_store, load_records


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("dataset")
    manifest = generate_dataset(root, scale="tiny")
    splits = {
        name: [json.loads(line) for line in (root / f"{name}.jsonl").read_text().splitlines()]
        for name in ("train", "valid", "test")
    }
    return root, manifest, splits


def test_manifest_hashes_and_case_contract(dataset):
    root, manifest, splits = dataset
    for split, cases in splits.items():
        assert len(cases) == manifest["counts"][split]
        assert len({(c["store_id"], c["question"]) for c in cases}) == len(cases)
        assert hashlib.sha256((root / f"{split}.jsonl").read_bytes()).hexdigest() == manifest["sha256"][split]
        for case in cases:
            assert (root / case["db_path"]).is_file()
            assert case["schema"] and case["policy"]
            if case["expected_action"] == "query":
                assert case["reference_sql"] and case["output_columns"]
                assert case["expected_rows"], "Nonempty gold rows prevent trivial empty-query acceptance"
                assert (
                    set(case["output_columns"]) == set(case["expected_rows"][0])
                    if case["expected_rows"]
                    else True
                )
            else:
                assert case["reference_sql"] is None
                assert case["expected_rows"] == []


def test_store_and_composition_split_boundaries(dataset):
    _, manifest, splits = dataset
    train_ids = {c["store_id"] for c in splits["train"]}
    valid_ids = {c["store_id"] for c in splits["valid"]}
    assert train_ids.isdisjoint(valid_ids)
    assert not set(HELDOUT_PATTERNS) & {c["pattern_id"] for c in splits["train"]}
    for case in splits["test"]:
        if case["slice"] in ("unseen_store", "unseen_both", "ambiguous"):
            assert case["store_id"] not in train_ids | valid_ids
        if case["slice"] in ("unseen_pattern", "unseen_both"):
            assert case["pattern_id"] in HELDOUT_PATTERNS
        if case["slice"] == "unseen_store":
            assert case["pattern_id"] in BASE_PATTERNS
    assert set(manifest["test_stores"]).isdisjoint(manifest["train_stores"])


def test_all_reference_queries_match_independent_oracle(dataset):
    root, _, splits = dataset
    for cases in splits.values():
        for case in cases:
            if not case["reference_sql"]:
                continue
            with sqlite3.connect(root / case["db_path"]) as db:
                db.row_factory = sqlite3.Row
                result = [dict(row) for row in db.execute(case["reference_sql"])]
            assert _equal(result, case["expected_rows"]), case["id"]


def test_oracle_catches_executable_but_wrong_queries(tmp_path: Path):
    path = tmp_path / "store.db"
    create_store(path, 42, "fixture", tiny=True)
    records = load_records(path)
    params = {"days": 14, "limit": 100, "threshold": 10, "horizon": 7}
    expected = oracle(records, "sales_units", params)
    correct_sql = build_sql("sales_units", params)
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        wrong_status = [dict(r) for r in db.execute(correct_sql.replace("o.status='paid' AND ", ""))]
        wrong_date = [
            dict(r)
            for r in db.execute(
                correct_sql.replace("o.created_at < '2026-10-01'", "o.created_at <= '2026-10-01'")
            )
        ]
    assert not _equal(expected, wrong_status)
    assert not _equal(expected, wrong_date)


def test_independent_oracle_responds_to_record_mutation(tmp_path: Path):
    path = tmp_path / "store.db"
    create_store(path, 42, "fixture", tiny=True)
    original = load_records(path)
    changed = copy.deepcopy(original)
    # Explicit fixture is last paid line before the final cancelled order.
    changed["order_items"][-3]["quantity"] += 10000
    params = {"days": 14, "limit": 100, "threshold": 10, "horizon": 7}
    assert not _equal(oracle(original, "sales_units", params), oracle(changed, "sales_units", params))


def test_partial_return_join_fanout_detected(tmp_path: Path):
    path = tmp_path / "store.db"
    create_store(path, 42, "fixture", tiny=True)
    expected = oracle(load_records(path), "net_revenue", {"days": 14, "limit": 100})
    # Joining raw returns directly duplicates original revenue when one line has multiple returns.
    wrong_sql = """SELECT i.variant_id,
    SUM(i.quantity*i.unit_price_cents-i.discount_cents-COALESCE(r.refund_cents,0)) AS net_revenue_cents
    FROM order_items i JOIN orders o ON o.order_id=i.order_id
    LEFT JOIN returns r ON r.item_id=i.item_id AND r.returned_at<'2026-10-01'
    WHERE o.status='paid' AND o.created_at>='2026-09-17' AND o.created_at<'2026-10-01'
    GROUP BY i.variant_id ORDER BY net_revenue_cents DESC,i.variant_id ASC LIMIT 100"""
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        wrong = [dict(row) for row in db.execute(wrong_sql)]
    assert not _equal(expected, wrong)
