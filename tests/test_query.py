import sqlite3

import pytest

from commerce_expert.query import QueryRejected, execute_readonly, parse_prediction, rows_equal


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "s.db"
    with sqlite3.connect(p) as c:
        c.execute("create table products(id integer, name text)")
        c.executemany("insert into products values(?, ?)", [(1, "A"), (2, "B")])
    return p


def test_read(db):
    assert execute_readonly(db, "with a as (select * from products) select count(*) as n from a") == [
        {"n": 2}
    ]


@pytest.mark.parametrize(
    "sql",
    [
        "delete from products",
        "select 1; drop table products",
        'attach database ":memory:" as x',
        "pragma table_info(products)",
        "select load_extension('x')",
        "select * from sqlite_master",
    ],
)
def test_reject(db, sql):
    with pytest.raises((QueryRejected, sqlite3.DatabaseError)):
        execute_readonly(db, sql)
    assert execute_readonly(db, "select count(*) as n from products") == [{"n": 2}]


def test_limit(db):
    with pytest.raises(QueryRejected):
        execute_readonly(db, "select * from products", max_rows=1)


def test_timeout(db):
    with pytest.raises(sqlite3.OperationalError):
        execute_readonly(
            db,
            "with recursive t(x) as (select 1 union all select x+1 from t) select sum(x) from t",
            timeout_seconds=0.01,
        )


def test_comparison():
    assert rows_equal([{"n": 1}, {"n": 2}], [{"n": 2}, {"n": 1}])
    assert not rows_equal([{"n": 1}, {"n": 1}], [{"n": 1}])
    assert not rows_equal([{"wrong": 1}], [{"n": 1}])


def test_parse():
    assert parse_prediction('```json\n{"action":"query","sql":"select 1"}\n```')["action"] == "query"
    with pytest.raises(QueryRejected):
        parse_prediction("select 1")


def test_comparison_accepts_equivalent_sql_numeric_types():
    assert rows_equal([{"n": 1.0, "rate": 0.333333333}], [{"n": 1, "rate": 0.333333}])
    assert not rows_equal([{"n": 1.01}], [{"n": 1}])
