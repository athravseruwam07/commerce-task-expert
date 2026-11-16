import sqlite3
from pathlib import Path

from commerce_expert.data import AS_OF_DATE, create_store, load_records


def test_seeded_store_is_reproducible(tmp_path: Path):
    left, right = tmp_path/'left.db', tmp_path/'right.db'
    assert create_store(left, 123, 'sample', tiny=True) == create_store(right, 123, 'sample', tiny=True)
    assert load_records(left) == load_records(right)
    assert left.read_bytes() == right.read_bytes()


def test_records_obey_commerce_constraints_and_include_edge_cases(tmp_path: Path):
    path = tmp_path/'store.db'
    create_store(path, 42, 'sample', tiny=True)
    records = load_records(path)
    for inventory in records['inventory']:
        assert 0 <= inventory['reserved'] <= inventory['on_hand']
    assert any(r['on_hand'] == 0 for r in records['inventory'])
    assert any(o['created_at'] == AS_OF_DATE for o in records['orders'])
    assert any(o['status'] == 'cancelled' for o in records['orders'])
    items = {i['item_id']: i for i in records['order_items']}
    quantity_by_item = {}
    count_by_item = {}
    for ret in records['returns']:
        assert ret['returned_at'] < AS_OF_DATE
        quantity_by_item[ret['item_id']] = quantity_by_item.get(ret['item_id'], 0)+ret['quantity']
        count_by_item[ret['item_id']] = count_by_item.get(ret['item_id'], 0)+1
    assert any(n > 1 for n in count_by_item.values())
    assert all(qty <= items[iid]['quantity'] for iid, qty in quantity_by_item.items())
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert not db.execute('PRAGMA foreign_key_check').fetchall()
