import json
import sqlite3

from commerce_expert.evaluate import evaluate_prediction, select_cases


def test_execution_is_not_answer_correctness(tmp_path):
    path = tmp_path / "s.db"
    with sqlite3.connect(path) as conn:
        conn.execute("create table stock(n integer)")
        conn.execute("insert into stock values(5)")
    case = {"db_path": "s.db", "expected_action": "query", "expected_rows": [{"n": 5}]}
    wrong = evaluate_prediction(
        case, {"text": json.dumps({"action": "query", "sql": "select 9 as n"})}, tmp_path
    )
    assert wrong["execution_success"] and not wrong["correct"]
    right = evaluate_prediction(
        case, {"text": json.dumps({"action": "query", "sql": "select n from stock"})}, tmp_path
    )
    assert right["correct"]


def test_clarification_decision(tmp_path):
    case = {"expected_action": "clarify"}
    result = evaluate_prediction(case, {"text": '{"action":"clarify","question":"Which period?"}'}, tmp_path)
    assert result["correct"] and not result["execution_success"]


def test_slice_selection():
    cases = [
        {"id": str(i), "slice": s, "pattern_id": p}
        for i, (s, p) in enumerate(
            [("new_store", "a"), ("new_store", "a"), ("new_pattern", "b"), ("both", "c")]
        )
    ]
    selected = select_cases(cases, 3)
    assert len(selected) == 3 and len({c["slice"] for c in selected}) == 3
