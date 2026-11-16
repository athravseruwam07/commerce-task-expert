# Dataset and business semantics

The corpus is entirely synthetic and deterministically generated from seed 42. There are 20 independent SQLite stores, each with 24 products, 48 variants, 1,600–2,300 random orders plus explicit boundary fixtures, line items, inventory snapshots, and partial returns. Prices and revenue are integer cents. Store seeds vary category allocation, item popularity, pricing, inventory, cancellation probability, and return probability. Generation takes seconds and requires no language-model API.

`commerce_expert.benchmark.generate_dataset(Path('data/generated'))` writes `stores/store_01.db` through `store_20.db`, JSONL cases, and a manifest with split SHA-256 hashes. The smaller `scale='tiny'` mode exists only for unit tests; published model results use `small`. Generated stores and data are excluded from Git; seeds, generation code and manifest checks make reconstruction possible.

## Schema and interpretation

- `products`: product identity, display name, category.
- `variants`: variant identity, parent product, SKU, current unit price.
- `inventory`: one as-of snapshot per variant, on-hand units and reservations.
- `orders`: creation date, paid or cancelled status.
- `order_items`: purchased quantity, historical unit price, total line discount.
- `returns`: individual partial return events, processed date, quantity, refund cents, restock flag.

The as-of date is **2026-10-01**. All timestamps are UTC calendar dates. Recent N days means `[as_of − N days, as_of)`, including the first date and excluding the as-of date. Paid orders alone count as sales. Line discounts are subtracted once per line. Gross sales exclude refunds, tax and shipping; net sales subtract linked refunds processed before the as-of date. A returned-item cohort is determined by order creation date; a report of return events instead uses return processing date.

Available stock is `on_hand − reserved`. Restocking is already reflected in the snapshot and must not be counted again. Paid sales units divided by lookback days estimate daily demand; returns do not reduce demand. Stockout risk means positive recent demand and available stock strictly below demand multiplied by the horizon. It includes already sold-out variants. This is a transparent constant-rate heuristic, **not a probability forecast**, and ignores inbound purchase orders and seasonal forecasting. Days of cover are rounded to four decimals.

## Splits and leakage control

| Split | Cases | Stores | Patterns |
|---|---:|---|---|
| Train | 2,000 | 01–12 | 11 basic query families plus clarification |
| Validation | 160 | 13–16 | Basic families with separate wording |
| Test: unseen store | 60 | 17–20 | Basic families with separate wording |
| Test: unseen pattern | 60 | 01–12 | Held-out compositions |
| Test: unseen both | 60 | 17–20 | Held-out compositions |
| Test: ambiguity | 60 | 17–20 | Separate underspecified questions |

The 11 training families cover variant unit sales, gross revenue, available stock, zero sales, returns by processing date, stockout coverage, product aggregation, category revenue, daily revenue, cohort return rate and net revenue. Held-out compositions combine low-stock filtering with sales ranking, aggregate cohort return rates by category, and compare recent sales with a preceding equal-length period. Their pattern IDs appear in neither training nor validation. Paraphrases of a held-out composition stay out of training.

Familiar-family test cases intentionally share basic SQL structures with training, because that slice isolates unseen-store generalization. Training has two substantive surface forms per basic family; validation and test each use a different complete wording. Ambiguous questions have separate wording by split and deliberately omit a decision metric or business objective. Complete `(store_id, question)` prompts are unique within each split. Case IDs, SQL, expected rows, split labels and parameters must never be included in model prompts. Model prompts contain only schema, policy, as-of date, and merchant question. Gold-derived output column names remain evaluator metadata and are never supplied to the model; otherwise their presence would leak the query-versus-clarification decision.

All stores share one schema. This benchmark tests **synthetic store-data and compositional generalization**, not unseen-schema transfer or performance on real merchant data. There are only three held-out composition families, so conclusions about broad SQL generalization must remain limited. No schema renaming or API-doc retrieval is included.

## Verification

Each reference query is executed and checked against an independent Python oracle that iterates raw records and computes joins, aggregations, cohorts, refunds and rankings without executing generated SQL. Expected rows come from that oracle, not from reference query output. Dataset generation fails immediately on any mismatch. Rows have deterministic metric/ID tie-breaking. Reference output aliases are evaluator metadata rather than part of the merchant request. All query cases have nonempty reference results, preventing a query that always returns no rows from passing on an empty gold result.

Every store includes zero-stock variants, a never-sold variant, cancelled orders, a paid order on the first day of a seven-day window, an excluded order on the as-of date, and one item with two partial returns. Unit tests verify reproducibility and show that executable queries with incorrect status filters, inclusive upper dates or raw-return join fanout produce incorrect results.

Exact answers use the complete result, not the demo's displayed row cap. Integer cents compare exactly; floating-point coverage and rates require a small tolerance. Scoring accepts one consistent output column permutation across the full result so equivalent aliases or column order do not cause false failures; required row ordering and multiplicity remain enforced. Clarification action accuracy is meaningful, but a generic clarification string alone does not prove a useful conversation; qualitative clarification quality should be audited separately.
