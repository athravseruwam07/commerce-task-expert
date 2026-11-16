"""Bounded, read-only SQLite execution and semantic result comparison."""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections import Counter
from pathlib import Path

import sqlglot
from sqlglot import exp


class QueryRejected(ValueError):
    pass


def parse_prediction(text: str) -> dict:
    """Require a structured decision; malformed output is an observed failure."""
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1]).strip() if lines[-1].strip() == "```" else raw
    try:
        value = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise QueryRejected("Model did not return valid JSON") from exc
    if not isinstance(value, dict) or value.get("action") not in {"query", "clarify"}:
        raise QueryRejected("Expected a query or clarify decision")
    if value["action"] == "query" and not isinstance(value.get("sql"), str):
        raise QueryRejected("Query decision requires SQL")
    if value["action"] == "clarify" and not isinstance(value.get("question"), str):
        raise QueryRejected("Clarify decision requires a question")
    return value


def validate_sql(sql: str) -> str:
    if not sql.strip() or len(sql) > 24000:
        raise QueryRejected("Empty or oversized SQL")
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except sqlglot.errors.ParseError as exc:
        raise QueryRejected("Invalid SQL syntax") from exc
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        raise QueryRejected("Only one SELECT query is allowed")
    forbidden = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Create,
        exp.Drop,
        exp.Command,
        exp.Alter,
        exp.Merge,
        exp.Into,
    )
    if any(isinstance(node, forbidden) for node in statements[0].walk()):
        raise QueryRejected("SQL contains a forbidden operation")
    return sql


def execute_readonly(
    db_path: Path, sql: str, timeout_seconds: float = 2.0, max_rows: int = 1000
) -> list[dict]:
    validate_sql(sql)
    path = Path(db_path).resolve()
    if not path.is_file():
        raise QueryRejected("Store database does not exist")
    started = time.monotonic()
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}

    def authorize(action, arg1, arg2, db, trigger):
        if action == sqlite3.SQLITE_FUNCTION and str(arg2).lower() in {
            "load_extension",
            "writefile",
            "readfile",
        }:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ and str(arg1).startswith("sqlite_"):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.set_authorizer(authorize)
        conn.set_progress_handler(lambda: int(time.monotonic() - started > timeout_seconds), 1000)
        cursor = conn.execute(sql)
        columns = [col[0] for col in cursor.description or []]
        if len(set(columns)) != len(columns):
            raise QueryRejected("Duplicate output column names; use distinct aliases")
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            raise QueryRejected(f"Result exceeds {max_rows} rows; narrow the question")
        return [dict(row) for row in rows]


def _normal(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        rounded = round(value, 6)
        return int(rounded) if rounded.is_integer() else rounded
    return value


def rows_equal(actual: list[dict], expected: list[dict], ordered: bool = False) -> bool:
    """Column names and multiplicity matter. Ignore order unless task requires it."""

    def canonical(rows):
        return [
            json.dumps(
                {key: _normal(value) for key, value in row.items()}, sort_keys=True, separators=(",", ":")
            )
            for row in rows
        ]

    left, right = canonical(actual), canonical(expected)
    return left == right if ordered else Counter(left) == Counter(right)


def results_equivalent(actual: list[dict], expected: list[dict], ordered: bool = False) -> bool:
    """SQL denotation equality under column renaming/permutation, not SQL text matching.

    The permutation is global across all rows, so columns cannot be swapped
    independently per row. Benchmark results contain at most four columns.
    """
    from itertools import permutations

    if len(actual) != len(expected):
        return False
    if not actual:
        return True
    actual_keys, expected_keys = list(actual[0]), list(expected[0])
    if len(actual_keys) != len(expected_keys) or len(expected_keys) > 6:
        return False
    if any(list(row) != actual_keys for row in actual) or any(list(row) != expected_keys for row in expected):
        return False
    for assignment in permutations(actual_keys):
        renamed = [
            {target: row[source] for source, target in zip(assignment, expected_keys)} for row in actual
        ]
        if rows_equal(renamed, expected, ordered):
            return True
    return False
