"""Verified question-to-SQL corpus with grouped, explicit generalization slices."""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from .data import AS_OF_DATE, POLICY, SCHEMA, create_store, load_records

BASE_PATTERNS = ("sales_units", "top_revenue", "low_stock", "no_sales", "return_units",
                 "stockout", "product_sales", "category_revenue", "daily_revenue", "return_rate", "net_revenue")
HELDOUT_PATTERNS = ("low_stock_bestsellers", "category_return_rate", "momentum")
AMBIGUOUS = ("Which products are performing best?", "What should I reorder?", "Which products need attention?",
             "Show me the worst products.", "How is my store doing?")


def _bounds(days: int) -> tuple[str, str]:
    end = date.fromisoformat(AS_OF_DATE)
    return (end - timedelta(days=days)).isoformat(), AS_OF_DATE


def build_sql(pattern: str, p: dict) -> str:
    start, end = _bounds(p["days"])
    filt = f"o.status='paid' AND o.created_at >= '{start}' AND o.created_at < '{end}'"
    joins = "order_items i JOIN orders o ON o.order_id=i.order_id"
    revenue = "i.quantity*i.unit_price_cents-i.discount_cents"
    limit = p["limit"]
    sales = (f"SELECT i.variant_id, SUM(i.quantity) AS units, SUM({revenue}) AS revenue_cents "
             f"FROM {joins} WHERE {filt} GROUP BY i.variant_id")
    rs = ("SELECT r.item_id,SUM(r.quantity) AS units,SUM(r.refund_cents) AS refunds "
          f"FROM returns r WHERE r.returned_at < '{end}' GROUP BY r.item_id")
    if pattern == "sales_units":
        return f"SELECT i.variant_id,SUM(i.quantity) AS units FROM {joins} WHERE {filt} GROUP BY i.variant_id ORDER BY units DESC,i.variant_id ASC LIMIT {limit}"
    if pattern == "top_revenue":
        return f"SELECT i.variant_id,SUM({revenue}) AS revenue_cents FROM {joins} WHERE {filt} GROUP BY i.variant_id ORDER BY revenue_cents DESC,i.variant_id ASC LIMIT {limit}"
    if pattern == "low_stock":
        return f"SELECT variant_id,on_hand-reserved AS available FROM inventory WHERE on_hand-reserved < {p['threshold']} ORDER BY available ASC,variant_id ASC"
    if pattern == "no_sales":
        return f"SELECT v.variant_id FROM variants v WHERE NOT EXISTS (SELECT 1 FROM {joins} WHERE i.variant_id=v.variant_id AND {filt}) ORDER BY v.variant_id ASC"
    if pattern == "return_units":
        return (f"SELECT i.variant_id,SUM(r.quantity) AS returned_units FROM returns r JOIN order_items i ON i.item_id=r.item_id JOIN orders o ON o.order_id=i.order_id WHERE o.status='paid' AND r.returned_at >= '{start}' AND r.returned_at < '{end}' GROUP BY i.variant_id ORDER BY returned_units DESC,i.variant_id ASC LIMIT {limit}")
    if pattern == "stockout":
        return (f"WITH sales AS ({sales}) SELECT inv.variant_id,inv.on_hand-inv.reserved AS available,s.units AS units,ROUND(1.0*(inv.on_hand-inv.reserved)*{p['days']}/s.units,4) AS days_cover FROM inventory inv JOIN sales s ON s.variant_id=inv.variant_id WHERE inv.on_hand-inv.reserved < 1.0*s.units*{p['horizon']}/{p['days']} ORDER BY days_cover ASC,inv.variant_id ASC")
    if pattern in ("product_sales", "category_revenue"):
        group, metric, alias = ("p.product_id", "i.quantity", "units") if pattern == "product_sales" else ("p.category", revenue, "revenue_cents")
        return f"SELECT {group},SUM({metric}) AS {alias} FROM {joins} JOIN variants v ON v.variant_id=i.variant_id JOIN products p ON p.product_id=v.product_id WHERE {filt} GROUP BY {group} ORDER BY {alias} DESC,{group} ASC LIMIT {limit}"
    if pattern == "daily_revenue":
        return f"SELECT o.created_at AS day,SUM({revenue}) AS revenue_cents FROM {joins} WHERE {filt} GROUP BY o.created_at ORDER BY day ASC"
    if pattern in ("return_rate", "category_return_rate"):
        group = "i.variant_id" if pattern == "return_rate" else "p.category"
        extra = "" if pattern == "return_rate" else " JOIN variants v ON v.variant_id=i.variant_id JOIN products p ON p.product_id=v.product_id"
        return (f"WITH ret AS ({rs}) SELECT {group},SUM(i.quantity) AS sold_units,SUM(COALESCE(ret.units,0)) AS returned_units,ROUND(1.0*SUM(COALESCE(ret.units,0))/SUM(i.quantity),4) AS return_rate FROM {joins} LEFT JOIN ret ON ret.item_id=i.item_id{extra} WHERE {filt} GROUP BY {group} ORDER BY return_rate DESC,{group} ASC LIMIT {limit}")
    if pattern == "net_revenue":
        return (f"WITH ret AS ({rs}) SELECT i.variant_id,SUM({revenue}-COALESCE(ret.refunds,0)) AS net_revenue_cents FROM {joins} LEFT JOIN ret ON ret.item_id=i.item_id WHERE {filt} GROUP BY i.variant_id ORDER BY net_revenue_cents DESC,i.variant_id ASC LIMIT {limit}")
    if pattern == "low_stock_bestsellers":
        return (f"WITH sales AS ({sales}) SELECT inv.variant_id,inv.on_hand-inv.reserved AS available,s.units FROM inventory inv JOIN sales s ON s.variant_id=inv.variant_id WHERE inv.on_hand-inv.reserved < {p['threshold']} ORDER BY s.units DESC,inv.variant_id ASC LIMIT {limit}")
    if pattern == "momentum":
        prior_start = (date.fromisoformat(start)-timedelta(days=p['days'])).isoformat()
        return (f"SELECT i.variant_id,SUM(CASE WHEN o.created_at >= '{start}' THEN i.quantity ELSE 0 END) AS recent_units,SUM(CASE WHEN o.created_at < '{start}' THEN i.quantity ELSE 0 END) AS prior_units,SUM(CASE WHEN o.created_at >= '{start}' THEN i.quantity ELSE -i.quantity END) AS unit_change FROM {joins} WHERE o.status='paid' AND o.created_at >= '{prior_start}' AND o.created_at < '{end}' GROUP BY i.variant_id ORDER BY unit_change DESC,i.variant_id ASC LIMIT {limit}")
    raise ValueError(f"Unknown pattern: {pattern}")


def oracle(records: dict, pattern: str, p: dict) -> list[dict]:
    """Independent record iteration; no generated SQL is consulted."""
    start, end = _bounds(p["days"])
    orders = {r['order_id']: r for r in records['orders']}
    items = {r['item_id']: r for r in records['order_items']}
    variants = {r['variant_id']: r for r in records['variants']}
    products = {r['product_id']: r for r in records['products']}
    inv = {r['variant_id']: r['on_hand']-r['reserved'] for r in records['inventory']}
    returns = defaultdict(lambda: [0, 0])
    for ret in records['returns']:
        if ret['returned_at'] < end:
            returns[ret['item_id']][0] += ret['quantity']
            returns[ret['item_id']][1] += ret['refund_cents']
    units, revenues, returned, net, daily = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    for item in records['order_items']:
        order = orders[item['order_id']]
        if order['status'] != 'paid' or not start <= order['created_at'] < end:
            continue
        vid, qty = item['variant_id'], item['quantity']
        rev = qty*item['unit_price_cents']-item['discount_cents']
        units[vid] += qty
        revenues[vid] += rev
        returned[vid] += returns[item['item_id']][0]
        net[vid] += rev-returns[item['item_id']][1]
        daily[order['created_at']] += rev
    k = p['limit']
    if pattern in ('sales_units', 'top_revenue', 'net_revenue'):
        values, col = {'sales_units': (units, 'units'), 'top_revenue': (revenues, 'revenue_cents'), 'net_revenue': (net, 'net_revenue_cents')}[pattern]
        return [{'variant_id': vid, col: value} for vid, value in sorted(values.items(), key=lambda x: (-x[1], x[0]))[:k]]
    if pattern == 'low_stock':
        return [{'variant_id': vid, 'available': value} for vid, value in sorted(inv.items(), key=lambda x: (x[1], x[0])) if value < p['threshold']]
    if pattern == 'no_sales':
        return [{'variant_id': vid} for vid in sorted(variants) if vid not in units]
    if pattern == 'return_units':
        result = defaultdict(int)
        for ret in records['returns']:
            item = items[ret['item_id']]
            if orders[item['order_id']]['status'] == 'paid' and start <= ret['returned_at'] < end:
                result[item['variant_id']] += ret['quantity']
        return [{'variant_id': vid, 'returned_units': n} for vid, n in sorted(result.items(), key=lambda x: (-x[1], x[0]))[:k]]
    if pattern == 'stockout':
        rows = [{'variant_id': vid, 'available': inv[vid], 'units': n, 'days_cover': round(inv[vid]*p['days']/n, 4)} for vid, n in units.items() if inv[vid] < n*p['horizon']/p['days']]
        return sorted(rows, key=lambda r: (r['days_cover'], r['variant_id']))
    if pattern in ('product_sales', 'category_revenue'):
        values = defaultdict(int)
        for vid, n in (units if pattern == 'product_sales' else revenues).items():
            pid = variants[vid]['product_id']
            group = pid if pattern == 'product_sales' else products[pid]['category']
            values[group] += n
        key, metric = ('product_id', 'units') if pattern == 'product_sales' else ('category', 'revenue_cents')
        return [{key: group, metric: n} for group, n in sorted(values.items(), key=lambda x: (-x[1], x[0]))[:k]]
    if pattern == 'daily_revenue':
        return [{'day': day, 'revenue_cents': n} for day, n in sorted(daily.items())]
    if pattern in ('return_rate', 'category_return_rate'):
        sold_group, ret_group = defaultdict(int), defaultdict(int)
        for vid, n in units.items():
            group = vid if pattern == 'return_rate' else products[variants[vid]['product_id']]['category']
            sold_group[group] += n
            ret_group[group] += returned[vid]
        key = 'variant_id' if pattern == 'return_rate' else 'category'
        rows = [{key: group, 'sold_units': n, 'returned_units': ret_group[group], 'return_rate': round(ret_group[group]/n, 4)} for group, n in sold_group.items()]
        return sorted(rows, key=lambda r: (-r['return_rate'], r[key]))[:k]
    if pattern == 'low_stock_bestsellers':
        rows = [{'variant_id': vid, 'available': inv[vid], 'units': n} for vid, n in units.items() if inv[vid] < p['threshold']]
        return sorted(rows, key=lambda r: (-r['units'], r['variant_id']))[:k]
    if pattern == 'momentum':
        prior_start = (date.fromisoformat(start)-timedelta(days=p['days'])).isoformat()
        recent, prior = defaultdict(int), defaultdict(int)
        for item in records['order_items']:
            order = orders[item['order_id']]
            if order['status'] == 'paid' and prior_start <= order['created_at'] < end:
                target = recent if order['created_at'] >= start else prior
                target[item['variant_id']] += item['quantity']
        rows = [{'variant_id': vid, 'recent_units': recent[vid], 'prior_units': prior[vid], 'unit_change': recent[vid]-prior[vid]} for vid in set(recent)|set(prior)]
        return sorted(rows, key=lambda r: (-r['unit_change'], r['variant_id']))[:k]
    raise ValueError(pattern)


def question(pattern: str, p: dict, split: str, rng: random.Random) -> str:
    d, k, t, h = p['days'], p['limit'], p['threshold'], p['horizon']
    templates = {
        'sales_units': f'Which {k} variants sold the most units in the last {d} days?',
        'top_revenue': f'List the top {k} variants by gross sales revenue in cents over the last {d} days.',
        'low_stock': f'Which variants have fewer than {t} available units? Sort by available stock ascending.',
        'no_sales': f'Which variants had no paid sales in the last {d} days?',
        'return_units': f'Which {k} variants had the most returned units processed in the last {d} days?',
        'stockout': f'Which variants will run out within {h} days based on paid unit sales during the last {d} days? Show available units, sold units, and days of cover rounded to 4 decimals, sorted by days of cover ascending.',
        'product_sales': f'Which {k} products sold the most units across all their variants in the last {d} days?',
        'category_revenue': f'Rank the top {k} categories by gross sales revenue in cents for the last {d} days.',
        'daily_revenue': f'Show daily gross sales revenue in cents for days with paid sales during the last {d} days, oldest first.',
        'return_rate': f'Rank the top {k} variants by unit return rate for paid orders in the last {d} days. Show sold units, returned units and return rate rounded to 4 decimals.',
        'net_revenue': f'Which {k} variants have the highest net sales revenue in cents for paid orders from the last {d} days, after associated refunds processed before the as-of date?',
        'low_stock_bestsellers': f'Among variants with fewer than {t} available units, which {k} sold the most units in the last {d} days? Include available stock and units sold.',
        'category_return_rate': f'Rank the top {k} categories by unit return rate for paid orders in the last {d} days. Include sold and returned units and return rate rounded to 4 decimals.',
        'momentum': f'Which {k} variants have the largest unit sales increase in the last {d} days compared with the preceding {d} days? Include recent units, prior units and unit change.',
    }
    alternate = {
        'sales_units': f'Rank {k} SKUs by quantity purchased on paid orders over the past {d} days.',
        'top_revenue': f'Over the past {d} days, identify {k} SKUs with the largest gross revenue, in cents.',
        'low_stock': f'Find SKU inventory below {t} units after reservations, lowest inventory first.',
        'no_sales': f'List SKU IDs without any paid-order units during the past {d} days.',
        'return_units': f'For returns dated within the past {d} days, rank {k} SKUs by returned quantity.',
        'stockout': f'Using the past {d} days as the demand window, find SKUs with less than {h} days of stock remaining. Include available, units sold and days_cover rounded to 4 decimals, earliest depletion first.',
        'product_sales': f'Combine all SKUs of each product and rank {k} product IDs by units purchased over the past {d} days.',
        'category_revenue': f'Combine paid-order sales from the past {d} days by category and return the {k} highest gross revenue totals in cents.',
        'daily_revenue': f'For each selling date in the past {d} days, calculate gross revenue in cents in chronological order.',
        'return_rate': f'For the past {d} days of paid orders, find {k} SKUs with the highest ratio of returned units to purchased units. Report both counts and the ratio rounded to 4 decimals.',
        'net_revenue': f'Rank {k} SKUs by revenue retained on paid orders placed in the past {d} days, deducting all linked refunds before the as-of date; use cents.',
        'low_stock_bestsellers': templates['low_stock_bestsellers'],
        'category_return_rate': templates['category_return_rate'],
        'momentum': templates['momentum'],
    }
    validation = {
        'sales_units': f'I need the {k} highest-volume variants for a {d}-day sales report. Count paid units.',
        'top_revenue': f'Create a {d}-day leaderboard of {k} variants by gross revenue in cents.',
        'low_stock': f'Report all variants whose unreserved stock is under {t}, with the smallest stock first.',
        'no_sales': f'Give me the variants that sold zero units in the most recent {d}-day window.',
        'return_units': f'Build a leaderboard of {k} variants with the most units coming back on return dates in the last {d} days.',
        'stockout': f'I need an inventory coverage report: look back {d} days and flag variants whose available stock cannot cover {h} days of average demand. Include available, units, days_cover rounded to 4 decimals; sort by lowest coverage.',
        'product_sales': f'Create a {d}-day product leaderboard by paid unit quantity, combining variants; keep {k} rows.',
        'category_revenue': f'Create a {d}-day category leaderboard by gross revenue cents, keeping {k} rows.',
        'daily_revenue': f'Create a chronological {d}-day report with each paid-sales date and its gross revenue cents; omit dates without sales.',
        'return_rate': f'Create a {d}-day paid-order cohort report of {k} variants with the greatest unit return fraction. Include sold_units, returned_units and return_rate rounded to 4 decimals.',
        'net_revenue': f'Create a {d}-day paid-order cohort leaderboard of {k} variants by net revenue cents, including all related refunds available as of now.',
        'low_stock_bestsellers': templates['low_stock_bestsellers'],
        'category_return_rate': templates['category_return_rate'],
        'momentum': templates['momentum'],
    }
    test = {
        'sales_units': f'Looking back {d} days, what are our {k} biggest movers by paid units at variant level?',
        'top_revenue': f'What {k} variants brought in the most gross sales money during the previous {d} days? Express amounts in cents.',
        'low_stock': f'What variants have less than {t} units left to sell once reserved units are removed? Show least available first.',
        'no_sales': f'Find the variants nobody bought on a paid order during the previous {d} days.',
        'return_units': f'Which {k} variants came back in the greatest quantities during the previous {d} days, measured by return processing date?',
        'stockout': f'At the selling pace observed over the previous {d} days, what variants would exhaust available stock in under {h} days? Return available, units and days_cover rounded to 4 decimals, soonest first.',
        'product_sales': f'What {k} products were bought in the greatest quantities during the previous {d} days, treating variants as the same product?',
        'category_revenue': f'Where did gross sales money come from by category during the previous {d} days? Give the {k} largest category amounts in cents.',
        'daily_revenue': f'How much gross sales money in cents came in on each date with paid sales during the previous {d} days? Present dates from earliest to latest.',
        'return_rate': f'Of variants bought on paid orders in the previous {d} days, what {k} had the largest share of units returned by the as-of date? Show sold_units, returned_units and return_rate rounded to 4 decimals.',
        'net_revenue': f'After taking away refunds known before the as-of date, which {k} variants kept the most revenue cents from paid orders in the previous {d} days?',
        'low_stock_bestsellers': f'What {k} variants are selling fastest by total units during the previous {d} days among those with under {t} unreserved units left? Include available and units.',
        'category_return_rate': f'Across categories, which {k} have the largest returned share of paid units purchased in the previous {d} days? Show sold_units, returned_units and return_rate rounded to 4 decimals, including returns before the as-of date.',
        'momentum': f'What {k} variants gained the most paid unit sales when comparing the previous {d} days to the {d}-day period immediately before it? Show recent_units, prior_units and unit_change.',
    }
    if split == 'valid':
        return validation[pattern]
    if split == 'test':
        return test[pattern]
    return rng.choice([templates[pattern], alternate[pattern]])


def _equal(a: list[dict], b: list[dict]) -> bool:
    if len(a) != len(b):
        return False
    for left, right in zip(a, b):
        if left.keys() != right.keys():
            return False
        for key in left:
            x, y = left[key], right[key]
            if isinstance(x, (float, int)) and isinstance(y, (float, int)):
                if abs(x-y) > 0.00011:
                    return False
            elif x != y:
                return False
    return True


def generate_dataset(output_dir: Path, seed: int = 42, scale: str = 'small') -> dict:
    """Build verified stores/cases. 'tiny' is for fast tests; 'small' is full corpus."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    tiny = scale == 'tiny'
    stores = []
    records, connections = {}, {}
    for n in range(1, 21):
        sid = f'store_{n:02d}'
        path = output_dir/'stores'/f'{sid}.db'
        stores.append(create_store(path, seed*100+n, sid, tiny=tiny))
        records[sid] = load_records(path)
        db = sqlite3.connect(path)
        db.row_factory = sqlite3.Row
        connections[sid] = db
    counts = {'train': 2000, 'valid': 160, 'test': 240} if not tiny else {'train': 55, 'valid': 22, 'test': 60}
    hashes, slice_counts = {}, {}
    try:
        for split, count in counts.items():
            cases, seen = [], set()
            for index in range(count):
                if split == 'train':
                    pool, family, slice_name = list(range(1, 13)), BASE_PATTERNS[index % len(BASE_PATTERNS)], 'train'
                    ambiguous = index % 15 == 14
                elif split == 'valid':
                    pool, family, slice_name = list(range(13, 17)), BASE_PATTERNS[index % len(BASE_PATTERNS)], 'validation_unseen_store'
                    ambiguous = index % 10 == 9
                else:
                    group = index % 4
                    ambiguous = group == 3
                    if group == 0:
                        pool, family, slice_name = list(range(17, 21)), BASE_PATTERNS[(index//4) % len(BASE_PATTERNS)], 'unseen_store'
                    elif group == 1:
                        pool, family, slice_name = list(range(1, 13)), HELDOUT_PATTERNS[(index//4) % len(HELDOUT_PATTERNS)], 'unseen_pattern'
                    elif group == 2:
                        pool, family, slice_name = list(range(17, 21)), HELDOUT_PATTERNS[(index//4) % len(HELDOUT_PATTERNS)], 'unseen_both'
                    else:
                        pool, family, slice_name = list(range(17, 21)), 'ambiguous', 'ambiguous'
                sid = f'store_{rng.choice(pool):02d}'
                if ambiguous:
                    family = 'ambiguous'
                    # Deduplicate complete prompts, including ambiguity examples.
                    prefixes = {'train': ['Please tell me: ', 'Please help: ', 'For my shop: ', 'I need advice: ', 'Merchant question: ', 'For our business: '], 'valid': ['I am wondering: ', 'For my review: '], 'test': ['For this review: ', 'Please investigate: ', 'Help me decide: ', 'Merchant review: ']}[split]
                    for _ in range(1000):
                        sid = f'store_{rng.choice(pool):02d}'
                        ambiguity_questions = {'train': AMBIGUOUS, 'valid': ('Which items are good?', 'What stock deserves priority?', 'What should my strategy be?', 'Tell me the best category.', 'Which products should I focus on?'), 'test': ('Which items should we promote?', 'Tell me our strongest products.', 'Which stock problems should I fix first?', 'What deserves more investment?', 'Which merchandise is doing badly?')}[split]
                        q = rng.choice(prefixes) + rng.choice(ambiguity_questions)
                        if (sid, q) not in seen:
                            break
                    sql, expected, params = None, [], {}
                else:
                    for _ in range(1000):
                        sid = f'store_{rng.choice(pool):02d}'
                        params = {'days': rng.choice([7, 10, 14, 21, 28, 30, 45, 60, 75, 90]), 'limit': rng.choice([3, 5, 8, 10]), 'threshold': rng.choice([3, 5, 8, 10, 15, 20, 25, 30, 40, 50]), 'horizon': rng.choice([3, 7, 10, 14])}
                        q = question(family, params, split, rng)
                        if (sid, q) not in seen:
                            break
                    sql = build_sql(family, params)
                    expected = oracle(records[sid], family, params)
                    actual = [dict(row) for row in connections[sid].execute(sql)]
                    if not _equal(expected, actual):
                        raise AssertionError(f'Oracle mismatch {sid}/{family}: {expected[:2]} != {actual[:2]}')
                if (sid, q) in seen:
                    raise AssertionError(f'Duplicate case prompt: {split}/{sid}/{q}')
                seen.add((sid, q))
                columns = [col[0] for col in connections[sid].execute(sql).description] if sql else []
                case = {'id': f'{split}_{index:04d}', 'question': q, 'store_id': sid,
                        'db_path': f'stores/{sid}.db', 'schema': SCHEMA, 'as_of_date': AS_OF_DATE,
                        'policy': POLICY, 'expected_action': 'clarify' if ambiguous else 'query',
                        'reference_sql': sql, 'expected_rows': expected, 'output_columns': columns,
                        'ordered': True, 'pattern_id': family, 'split': split, 'slice': slice_name,
                        'parameters': params}
                if ambiguous:
                    case['reference_clarification'] = 'Which metric, time period, and ranking criteria should I use?'
                cases.append(case)
            payload = ''.join(json.dumps(case, sort_keys=True)+'\n' for case in cases)
            (output_dir/f'{split}.jsonl').write_text(payload)
            hashes[split] = hashlib.sha256(payload.encode()).hexdigest()
            sc = defaultdict(int)
            for case in cases:
                sc[case['slice']] += 1
            slice_counts[split] = dict(sc)
    finally:
        for db in connections.values():
            db.close()
    manifest = {'version': 1, 'seed': seed, 'scale': scale, 'as_of_date': AS_OF_DATE, 'stores': stores,
                'counts': counts, 'sha256': hashes, 'slice_counts': slice_counts,
                'train_stores': [f'store_{n:02d}' for n in range(1, 13)],
                'valid_stores': [f'store_{n:02d}' for n in range(13, 17)],
                'test_stores': [f'store_{n:02d}' for n in range(17, 21)],
                'train_patterns': list(BASE_PATTERNS), 'heldout_patterns': list(HELDOUT_PATTERNS),
                'scope': 'Fixed-schema synthetic data and compositional generalization; not unseen-schema evaluation.'}
    (output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest
