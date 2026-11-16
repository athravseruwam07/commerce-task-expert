# Measured experiment results

All values below come from actual local MLX runs. No SQL repair or oracle substitution.

| Variant | Cases | Overall correctness | Query-only correctness | Query execution | p50 generation | p95 generation | MLX peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 60 | 0.0% | 0.0% | 15.6% | 6.82s | 17.82s | 3.29 GB |
| fine_tuned | 60 | 13.3% | 17.8% | 22.2% | 5.07s | 18.38s | 3.29 GB |
| quantized | 60 | 10.0% | 13.3% | 24.4% | 2.05s | 6.08s | 1.31 GB |

Answer correctness includes correct clarify decisions. Query-only accuracy and individual slices are in summary.json. Clarification wording is not automatically graded. Peak MLX allocations are unified-memory accelerator allocations, not whole-process RAM. Generation time excludes model loading and SQL execution.

## Generalization slices

| Variant | Slice | Cases | Correct |
|---|---|---:|---:|
| base | ambiguous | 15 | 0.0% |
| base | unseen_both | 15 | 0.0% |
| base | unseen_pattern | 15 | 0.0% |
| base | unseen_store | 15 | 0.0% |
| fine_tuned | ambiguous | 15 | 0.0% |
| fine_tuned | unseen_both | 15 | 0.0% |
| fine_tuned | unseen_pattern | 15 | 0.0% |
| fine_tuned | unseen_store | 15 | 53.3% |
| quantized | ambiguous | 15 | 0.0% |
| quantized | unseen_both | 15 | 0.0% |
| quantized | unseen_pattern | 15 | 0.0% |
| quantized | unseen_store | 15 | 40.0% |

## Paired comparison

- base_to_fine_tuned: +13.3% absolute correctness change; task bootstrap 95% interval [+5.0%, +21.7%]; 8 wins, 0 losses.
- fine_tuned_to_quantized: -3.3% absolute correctness change; task bootstrap 95% interval [-8.3%, +0.0%]; 0 wins, 2 losses.

Intervals resample cases, which are correlated by store and pattern; they are descriptive and should not be interpreted as independent evidence across hundreds of novel SQL structures.

## Failure analysis

### base

Failures: {'execution_or_format': 53, 'wrong_result': 7}

- `test_0003` (ambiguous): For this review: Which stock problems should I fix first?
  - Error: QueryRejected: Model did not return valid JSON
  - Generated SQL: `None`
- `test_0006` (category_return_rate): Across categories, which 8 have the largest returned share of paid units purchased in the previous 28 days? Show sold_units, returned_units and return_rate rounded to 4 decimals, including returns before the as-of date.
  - Error: OperationalError: no such column: oi.returned_at
  - Generated SQL: `SELECT p.category, SUM(oi.quantity) AS sold_units, SUM(oi.quantity - oi.returned_at) AS returned_units, CAST(SUM(oi.quantity - oi.returned_at) AS REAL) / SUM(oi.quantity) AS return_rate, SUM(oi.quantity - oi.returned_at) AS return_monetary_value FROM products p JOIN variants v ON p.product_id = v.product_id JOIN order_items oi ON v.variant_id = oi.variant_id JOIN orders o ON oi.order_id = o.order_id WHERE o.created_at < '2026-10-01' AND o.status = 'paid' AND o.created_at >= '2026-09-03' GROUP BY p.category ORDER BY return_rate DESC, p.category DESC LIMIT 8`
- `test_0005` (category_return_rate): Across categories, which 5 have the largest returned share of paid units purchased in the previous 10 days? Show sold_units, returned_units and return_rate rounded to 4 decimals, including returns before the as-of date.
  - Error: OperationalError: no such column: oi.returned_at
  - Generated SQL: `SELECT p.category, SUM(oi.quantity) AS sold_units, SUM(oi.quantity - oi.returned_at) AS returned_units, CAST(SUM(oi.quantity - oi.returned_at) AS REAL) / SUM(oi.quantity) AS return_rate, SUM(oi.quantity - oi.returned_at) AS return_monetary_value FROM products p JOIN variants v ON p.product_id = v.product_id JOIN order_items oi ON v.variant_id = oi.variant_id JOIN orders o ON oi.order_id = o.order_id WHERE o.created_at >= DATE('now', '-10 days') AND o.status = 'paid' GROUP BY p.category ORDER BY return_rate DESC, p.category ASC LIMIT 5`

### fine_tuned

Failures: {'wrong_decision': 2, 'execution_or_format': 48, 'wrong_result': 2}

- `test_0003` (ambiguous): For this review: Which stock problems should I fix first?
  - Error: Executable query returned wrong rows or decision.
  - Generated SQL: `SELECT i.variant_id,SUM(i.quantity) AS units, SUM(i.quantity*i.unit_price_cents-i.discount_cents) AS revenue_cents FROM order_items i JOIN orders o ON o.order_id=i.order_id WHERE o.status='paid' AND o.created_at >= '2026-09-01' AND o.created_at < '2026-10-01' GROUP BY i.variant_id ORDER BY revenue_cents DESC,i.variant_id ASC`
- `test_0006` (category_return_rate): Across categories, which 8 have the largest returned share of paid units purchased in the previous 28 days? Show sold_units, returned_units and return_rate rounded to 4 decimals, including returns before the as-of date.
  - Error: QueryRejected: Model did not return valid JSON
  - Generated SQL: `None`
- `test_0005` (category_return_rate): Across categories, which 5 have the largest returned share of paid units purchased in the previous 10 days? Show sold_units, returned_units and return_rate rounded to 4 decimals, including returns before the as-of date.
  - Error: QueryRejected: Model did not return valid JSON
  - Generated SQL: `None`

### quantized

Failures: {'execution_or_format': 49, 'wrong_result': 5}

- `test_0003` (ambiguous): For this review: Which stock problems should I fix first?
  - Error: OperationalError: no such column: i.quantity
  - Generated SQL: `SELECT i.variant_id,SUM(i.quantity) AS units, SUM(COALESCE(COALESCE(r.quantity,0),0)) AS returned FROM inventory i LEFT JOIN returns r ON r.item_id=i.variant_id WHERE r.returned_at < '2026-10-01' GROUP BY i.variant_id ORDER BY units-desc,i.variant_id ASC`
- `test_0006` (category_return_rate): Across categories, which 8 have the largest returned share of paid units purchased in the previous 28 days? Show sold_units, returned_units and return_rate rounded to 4 decimals, including returns before the as-of date.
  - Error: OperationalError: no such column: p.category
  - Generated SQL: `SELECT p.category,SUM(i.quantity) AS sold_units, SUM(COALESCE(r.quantity,0)) AS returned_units,ROUND(1.0*SUM(COALESCE(r.quantity,0))/SUM(i.quantity),4) AS return_rate FROM order_items i JOIN orders o ON o.order_id=i.order_id LEFT JOIN returns r ON r.item_id=i.item_id WHERE o.status='paid' AND o.created_at >= '2026-09-03' AND o.created_at < '2026-10-01' GROUP BY p.category ORDER BY return_rate DESC,p.category ASC LIMIT 8`
- `test_0005` (category_return_rate): Across categories, which 5 have the largest returned share of paid units purchased in the previous 10 days? Show sold_units, returned_units and return_rate rounded to 4 decimals, including returns before the as-of date.
  - Error: OperationalError: no such column: p.category
  - Generated SQL: `SELECT p.category,SUM(i.quantity) AS sold_units, SUM(COALESCE(r.quantity,0)) AS returned_units,ROUND(1.0*SUM(COALESCE(r.quantity,0))/SUM(i.quantity),4) AS return_rate FROM order_items i JOIN orders o ON o.order_id=i.order_id LEFT JOIN returns r ON r.item_id=i.item_id WHERE o.status='paid' AND o.created_at >= '2026-09-21' AND o.created_at < '2026-10-01' GROUP BY p.category ORDER BY return_rate DESC,p.category ASC LIMIT 5`

## Limits

- Synthetic, shared-schema stores; this evaluates data and composition shifts, not arbitrary schema transfer.
- Small finite pattern library and templated wording limit the breadth of the claim.
- One short adaptation run, one training seed and one local machine; no universal speed or quality claim.
- Fixed variant order on a normal desktop workload; no randomized or thermally isolated latency study.
- The primary metric expects the reference output shape. Extra descriptive columns or SKU-versus-ID representations can be scored as mismatches; see docs/FAILURES.md for a qualitative audit.
- Stock risk is constant recent-sales velocity, not a probabilistic forecast; inventory-censored demand is not corrected.
- Result comparison allows a global column permutation and renaming; expected columns never enter model inputs.
- The explanation is deterministic evidence rendering; the learned component chooses query or clarification.
- Test outputs must not be used to tune the published model or prompt.
