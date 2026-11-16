# Failure analysis: observed model behavior

This document examines actual completed records in `reports/records/base.jsonl`. It is a qualitative audit of selected baseline cases, not an aggregate result or proof of fine-tuning gains. The model, prompt, corpus and evaluator are frozen throughout the comparison. See `reports/RESULTS.md` for the measured comparison.

## Executable SQL can answer the wrong question

### `test_0028`: one missing day changes the ranking

The merchant requests gross category revenue for the previous seven days as of **2026-10-01**. The policy defines a half-open interval, so the correct range is **2026-09-24 inclusive through 2026-10-01 exclusive**. The baseline produces a valid paid-order join and the correct line-revenue arithmetic, but starts one day too late:

```sql
WHERE o.created_at < '2026-10-01'
  AND o.status = 'paid'
  AND o.created_at >= '2026-09-25'
```

The query executes. It ranks Accessories first at **454,140 cents**, while the oracle ranks Apparel first at **483,190 cents**. The missing day changes both totals and their ranking. This is a semantic date-range failure, not a syntax error. Its `gross_sales` alias is acceptable: scoring ignores output aliases using one consistent column mapping.

### `test_0032`: wall-clock time overrides the supplied as-of date

The merchant requests daily revenue over the previous 60 days, oldest first. Correct dates are **2026-08-02 inclusive to 2026-10-01 exclusive**. The baseline instead generates:

```sql
WHERE T1.status = 'paid'
  AND T1.created_at >= DATE('now', '-60 days')
GROUP BY T1.created_at
ORDER BY T1.created_at ASC
```

This uses the execution machine's current date and omits the exclusive upper boundary. In the recorded run, it returns dates starting **2026-09-17** and includes the explicit **2026-10-01** boundary fixture, which contributes **6,047 cents**. The oracle excludes that day. The observed result is reproducible from the saved rows even though rerunning `date('now')` on another day could produce a different wrong answer. Merchant analytics should anchor relative dates to the supplied analysis date, not the machine clock.

## Schema grounding and return accounting

### `test_0005` and `test_0006`: inventing a return column

These held-out compositions ask for category return rates for orders placed in the previous 10 and 28 days, respectively. Both raw outputs include this expression:

```sql
SUM(oi.quantity - oi.returned_at) AS returned_units
```

SQLite rejects both with `no such column: oi.returned_at`. The supplied schema puts `returned_at` on **returns**, not **order_items**. A date would not be a quantity even if that column existed. `test_0005` additionally uses `DATE('now', '-10 days')` instead of the frozen analysis interval.

The required computation joins products to variants, variants to order items, and order items to paid orders. Separately aggregate eligible returns by `item_id` before joining them to sold lines, because one line can have multiple partial returns. Select paid orders by creation-date cohort, count linked return units processed before the as-of date, and divide returned units by sold units. A left join retains sold items with no returns. Aggregating raw return events directly with original line quantities can duplicate the denominator; this is why the stores contain explicit partial-return fixtures and the tests include a join-fanout counterexample.

These cases contain fenced JSON, which the parser accepts. Their observed rejection is therefore a schema error, not merely failure to obey the no-Markdown formatting instruction.

### `test_0029`: invalid business arithmetic can still execute

Another category-return query replaces return accounting with:

```sql
SUM(oi.quantity - oi.discount_cents / 100) AS returned_units
```

It never reads the returns table. It subtracts money from unit quantity and produces negative returned units and a negative return rate. For example, the recorded Accessories row has **−322 returned units** and a rate of approximately **−2.9009**. This illustrates a distinct error from invented columns: SQL execution succeeds, but dimensional consistency and business meaning fail.

## Clarification and output format are separate skills

### `test_0003`: an underspecified request becomes an unfinished query

The merchant asks: **“Which stock problems should I fix first?”** The policy does not choose the merchant's objective, priority criterion or planning horizon. The correct action is to ask a short clarification, such as whether to prioritize projected stockouts over a specified period.

The baseline instead emits a long prose plan, describes an unsupported stock-risk formula, and begins a SQL block. The saved output reaches the **512-token cap** and ends at `(SUM`. Parsing fails with `Model did not return valid JSON`; no SQL is executed. There are two distinct observations:

1. The response does not satisfy the structured action protocol and is incomplete at the configured token cap.
2. Its attempted answer makes business decisions that the merchant did not specify.

A protocol failure prevents automatic scoring of any SQL within that response. It does not prove that the model lacks every underlying reasoning capability. Conversely, producing valid JSON would not establish correct clarification behavior or correct SQL. The final report should keep protocol validity, query execution, answer correctness and clarification decision accuracy separate. A nonempty clarification string receives action-decision credit, but its usefulness requires human inspection.

## Reference-result shape: a scoring limitation

The scorer compares query results against a fixed oracle projection. It accepts different aliases and one consistent permutation of output columns while preserving row order, multiplicity and numeric requirements. It does not establish general SQL equivalence or automatically recognize all merchant-useful alternative representations.

For example, a correct result augmented with an extra descriptive column can fail a fixed projection check. A SKU-based answer might be meaningful to a merchant where a reference answer uses a variant ID; equivalence would require a schema-aware entity mapping, which this evaluator does not provide. Scalar summaries, different grouping granularity or omitted requested evidence can also produce a useful-looking answer without meeting the benchmark's complete requested result.

The inspected records do not justify treating the concrete failures above as successes hidden by aliases. `test_0028` and `test_0032` have wrong dates and wrong values. `test_0005` and `test_0006` do not execute. `test_0029` has invalid arithmetic. Another inspected case, `test_0040`, adds product names and extra statistics, but also aggregates at product rather than variant level, omits refund subtraction and uses the wrong date boundary; its failure is not solely a projection issue. `test_0012` selects SKU strings rather than IDs, but its `oi.quantity = 0` condition also returns no rows instead of finding variants without sales.

More broadly, finite database denotation is evidence, not a proof of query semantics. A logically wrong query can coincidentally match one store's result. Independent Python oracles verify the reference calculations; nonempty results, date-boundary fixtures, partial returns and multiple stores make some mistakes observable, but cannot eliminate every accidental match. A stronger future evaluator would run each candidate query against multiple counterfactual database instances and audit semantically equivalent output shapes separately.

## Follow-up experiments, without changing this test run

- **Few-shot baseline ablation:** compare the untouched model using the current fixed prompt against the same model given verified training-only demonstrations of JSON output, relative-date arithmetic and clarification. This tests whether an apparent fine-tuning gain mainly reflects protocol learning. Report extra prompt tokens and latency. Freeze a new protocol and fresh held-out task cohort before evaluating this follow-up; do not optimize demonstrations against the published test failures.
- **Error decomposition:** distinguish malformed/truncated output, invalid identifiers, runtime errors, wrong clarification decisions, wrong date windows, join fanout and wrong business arithmetic. Retain raw output so failures can be independently checked.
- **Clarification audit:** inspect whether the question asks for the actual missing objective or horizon, rather than merely returning a generic string.
- **Counterfactual verification:** evaluate the same generated query across stores modified at date boundaries or with additional partial returns. Correct query behavior should follow the documented business rules under those changes.
- **Output-shape audit:** separately review otherwise correct queries that only add descriptive columns or use SKU/name representations, without changing the frozen primary score retroactively.

These are proposed follow-ups, not completed experiments or claimed improvements.

## What the measured fine-tuned model changed

The completed `fine_tuned.jsonl` run supplies a controlled contrast on the same questions and prompt:

- On `test_0032`, the fine-tuned checkpoint generates the correct **2026-08-02 to 2026-10-01** window, joins paid orders correctly, and matches all 60 daily rows. The baseline uses the machine clock instead.
- On `test_0008`, the fine-tuned checkpoint returns the correct available-inventory filter and ordering. The baseline repeats unrelated revenue expressions until its output is truncated.
- On `test_0028`, the fine-tuned checkpoint fixes the seven-day start date to **2026-09-24**, but references `o.status` without joining the orders table. Its improvement in date arithmetic is insufficient for an executable answer.
- On `test_0006`, the fine-tuned model repeats nested `COALESCE` expressions and fails the JSON protocol. Training does not resolve this held-out category/return-rate composition.

All eight correct answers are in the familiar-pattern/unseen-store slice. The two held-out-composition slices and clarification slice have no correct answers for this checkpoint. This is evidence of limited generalization in this experiment, not proof of a particular internal memorization mechanism.

Checkpoint 200 had the lowest completion loss on the fixed 40 validation cases, **0.069**, but reaches only **8/60** correct held-out decisions/results. Teacher-forced validation loss and free-generated task correctness measure different behavior on different distributions. Limited training, distribution shift and autoregressive decoding errors are plausible explanations; this experiment does not isolate their individual effects. A follow-up should select checkpoints using generated validation-query correctness as well as loss, use a broader training curriculum, and keep a new final test cohort sealed.


## Quantization outcome

The quantized checkpoint retains six of the eight fine-tuned correct answers and gains no new correct cases. Overall correctness drops from 13.3% to 10.0%, while measured peak MLX memory falls from 3.29 GB to 1.31 GB and median generation time from 5.07 s to 2.05 s. This is an efficiency/quality tradeoff on this workload, not a general claim that 4-bit conversion always speeds inference or preserves SQL behavior. See the paired records and bootstrap analysis for exact cases and uncertainty.
