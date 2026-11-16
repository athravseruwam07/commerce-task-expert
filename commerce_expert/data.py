"""Deterministic synthetic merchant stores. All prices use integer cents."""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

AS_OF_DATE = "2026-10-01"
SCHEMA = """CREATE TABLE products (
 product_id INTEGER PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL);
CREATE TABLE variants (
 variant_id INTEGER PRIMARY KEY, product_id INTEGER REFERENCES products(product_id),
 sku TEXT NOT NULL, price_cents INTEGER NOT NULL);
CREATE TABLE inventory (
 variant_id INTEGER PRIMARY KEY REFERENCES variants(variant_id),
 on_hand INTEGER NOT NULL, reserved INTEGER NOT NULL);
CREATE TABLE orders (
 order_id INTEGER PRIMARY KEY, created_at TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('paid','cancelled')));
CREATE TABLE order_items (
 item_id INTEGER PRIMARY KEY, order_id INTEGER REFERENCES orders(order_id),
 variant_id INTEGER REFERENCES variants(variant_id), quantity INTEGER NOT NULL,
 unit_price_cents INTEGER NOT NULL, discount_cents INTEGER NOT NULL);
CREATE TABLE returns (
 return_id INTEGER PRIMARY KEY, item_id INTEGER REFERENCES order_items(item_id),
 returned_at TEXT NOT NULL, quantity INTEGER NOT NULL, refund_cents INTEGER NOT NULL,
 restocked INTEGER NOT NULL);
"""
POLICY = """SQLite dialect. Each database is one merchant; do not filter by store_id.
All dates are UTC calendar dates (YYYY-MM-DD). The as-of date is exclusive: recent N days means [as_of_date minus N days, as_of_date). Only status='paid' orders count as sales. Order item discount_cents is the total line discount, not per unit. Gross sales revenue = quantity * unit_price_cents - discount_cents, before returns, excluding tax/shipping. Net sales revenue subtracts refund_cents for returns processed before as_of_date against those order items. Available inventory = on_hand - reserved. Inventory is an as-of snapshot; restocked returns are already reflected in on_hand. Sales velocity is paid units / lookback days; do not subtract returns from demand. A stockout risk means positive recent sales and available inventory < sales_velocity * horizon_days, including already sold-out variants. This is a constant-rate estimate, not a calibrated probability. Return rate is returned units / paid units for orders in the lookback period, counting associated returns processed before as_of_date. Return units by return date instead uses returned_at in the requested period. Group by IDs as well as display names. Ties: metric descending, ID ascending unless another order is requested. Return monetary values in integer cents. Never guess an unspecified ranking metric. Queries must be read-only and reference only the supplied tables."""


def create_store(path: Path, seed: int, store_id: str, tiny: bool = False) -> dict:
    """Create a complete independent store; returns summary, never uses wall clock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    rng = random.Random(seed)
    as_of = date.fromisoformat(AS_OF_DATE)
    categories = ["Apparel", "Home", "Outdoors", "Accessories"]
    products, variants, inventory, orders, items, returns = [], [], [], [], [], []
    product_count = 12 if tiny else 24
    variant_count = product_count * 2
    for pid in range(1, product_count + 1):
        category = categories[(pid + seed) % len(categories)]
        products.append((pid, f"{category} Item {pid:02d}", category))
        for size in range(2):
            vid = (pid - 1) * 2 + size + 1
            price = rng.randrange(8, 125) * 100
            variants.append((vid, pid, f"{store_id.upper()}-{vid:03d}", price))
            hand = 0 if vid % 11 == 0 else rng.randint(2, 180)
            reserved = rng.randint(0, min(hand, 10))
            inventory.append((vid, hand, reserved))
    # Each store varies in demand, cancellation rate, returns and price.
    n_orders = 350 if tiny else rng.randint(1600, 2300)
    weights = [((variant_count - i) ** rng.uniform(0.5, 1.4)) for i in range(variant_count)]
    weights[-1] = 0  # deliberate never-sold SKU
    cancel_probability = rng.uniform(0.03, 0.15)
    return_probability = rng.uniform(0.06, 0.20)
    for oid in range(1, n_orders + 1):
        days_ago = rng.randint(1, 180)
        when = (as_of - timedelta(days=days_ago)).isoformat()
        status = "cancelled" if rng.random() < cancel_probability else "paid"
        orders.append((oid, when, status))
        chosen = rng.choices(range(1, variant_count + 1), weights=weights, k=rng.randint(1, 3))
        for vid in chosen:
            qty = rng.randint(1, 4)
            price = variants[vid - 1][3]
            discount = int(price * qty * rng.choice([0, 0, 0.1, 0.2]))
            iid = len(items) + 1
            items.append((iid, oid, vid, qty, price, discount))
            if status == "paid" and rng.random() < return_probability:
                returned = rng.randint(1, qty)
                when_returned = date.fromisoformat(when) + timedelta(days=rng.randint(1, 30))
                if when_returned < as_of:
                    refund = ((price * qty - discount) * returned) // qty
                    returns.append(
                        (
                            len(returns) + 1,
                            iid,
                            when_returned.isoformat(),
                            returned,
                            refund,
                            int(rng.random() < 0.8),
                        )
                    )
    # Boundary fixtures: first included day, exclusive as-of day, cancelled high revenue,
    # two separate partial returns for a single line (exposes join fan-out bugs).
    for days_ago, status in [(7, "paid"), (0, "paid"), (1, "cancelled")]:
        oid = len(orders) + 1
        orders.append((oid, (as_of - timedelta(days=days_ago)).isoformat(), status))
        iid = len(items) + 1
        items.append((iid, oid, 1, 5, 1234, 123))
        if days_ago == 7:
            for returned_ago in (3, 1):
                returns.append(
                    (len(returns) + 1, iid, (as_of - timedelta(days=returned_ago)).isoformat(), 1, 1209, 1)
                )
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA)
        db.executemany("INSERT INTO products VALUES (?,?,?)", products)
        db.executemany("INSERT INTO variants VALUES (?,?,?,?)", variants)
        db.executemany("INSERT INTO inventory VALUES (?,?,?)", inventory)
        db.executemany("INSERT INTO orders VALUES (?,?,?)", orders)
        db.executemany("INSERT INTO order_items VALUES (?,?,?,?,?,?)", items)
        db.executemany("INSERT INTO returns VALUES (?,?,?,?,?,?)", returns)
        db.executescript(
            "CREATE INDEX idx_orders_date ON orders(created_at); "
            "CREATE INDEX idx_items_variant ON order_items(variant_id); "
            "CREATE INDEX idx_returns_item ON returns(item_id);"
        )
    return {
        "store_id": store_id,
        "seed": seed,
        "products": len(products),
        "variants": len(variants),
        "orders": len(orders),
        "items": len(items),
        "returns": len(returns),
    }


def load_records(path: Path) -> dict[str, list[dict]]:
    """The oracle reads records, then computes using Python without aggregate SQL."""
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        return {
            table: [dict(row) for row in db.execute(f"SELECT * FROM {table}")]
            for table in ("products", "variants", "inventory", "orders", "order_items", "returns")
        }
