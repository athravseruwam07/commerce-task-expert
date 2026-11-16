# Commerce Task Expert — end-to-end plan

## Research question
Can supervised local fine-tuning improve a small open model's commerce SQL correctness, and how much improvement survives 4-bit quantization?

## Constraints and scope
All generation, training, quantization, and inference run locally on an Apple M4 MacBook Pro with 16 GB unified memory. SQLite is the single supported dialect. Synthetic stores only. Initial model: Qwen2.5-Coder-1.5B-Instruct. Initial training: LoRA with compact context and batch size 1, not full parameter training. No cloud GPU or paid teacher is required. GraphQL, real Shopify integration, multi-agent query repair, probabilistic forecasting, and production authentication are deferred.

## Milestones and acceptance criteria
1. **Database and business semantics.** Seeded stores with product variants, orders, inventory, and line-level returns. Explicit UTC date boundaries, cents, order-status policy, and stock-risk definition. Acceptance: deterministic rebuild, relational invariants and adversarial boundary tests.
2. **Verified benchmark.** Reference SQL independently checked with Python calculations. Group split by pattern and store; hold out SQL compositions, not only paraphrases. Acceptance: reference passes every oracle; no held-out patterns/stores in training, checksummed dataset manifest.
3. **Baseline.** Pin model revision and runtime; establish first-attempt greedy model results before tuning. Acceptance: raw model output, executed rows, correctness, errors, memory and generation latency recorded for each case.
4. **Fine-tuning.** Train question/schema/policy to JSON query-or-clarify response. Tune only using training and validation, select checkpoint on validation. Acceptance: reproducible config, adapter, fused model, actual training log and dataset hashes. Test never used to choose weights or prompts.
5. **Quantization.** Quantize the fused fine-tuned checkpoint to MLX 4-bit, retaining tokenizer and prompt. Acceptance: base / fine-tuned / quantized evaluated on identical immutable cases and runtime. Compare actual model bytes and memory separately.
6. **Demo.** Merchant question, model SQL, read-only results, deterministic evidence rendering, assumptions, latency, and failure display. Acceptance: real inference locally and clearly labeled replay of recorded experiments.
7. **Release.** Automated tests, pinned dependencies, model/data documentation, result report, failure analysis, reproducible commands, private GitHub repository and release assets for local checkpoints.

## Experimental controls
Use the same schema context, output-column contract, tokenizer/chat template, token cap, greedy sampler, single-request concurrency, and frozen as-of date across the three variants. Warm each model on validation, then evaluate in separate processes. No query repair in the primary comparison. Report execution and answer correctness separately. Ambiguous requests have a separate decision metric and a manual wording audit; decision correctness does not prove the clarification question is useful. Deterministic prose is not evaluated as learned language generation.

Use a small, deterministic, stratified pilot if a complete evaluation exceeds practical local runtime. Clearly disclose its size, IDs, and split coverage. Freeze selection before viewing test outputs. Expand to full held-out suite without changing model or prompt. Failed or slower quantization is a valid result. Do not claim probabilistic stockout forecasts or schema generalization from shared-schema stores.

## Risk management
Prevent test leakage by grouping all paraphrases under a pattern ID and assigning holdout stores before training data generation. Avoid falsely accepting equivalent-on-one-database wrong queries with independent Python oracles, nonempty result fixtures, edge cases, and cross-store evaluation. Enforce read-only mode, SQLite authorizer, row limits, and query timeout. Profile a training smoke run before spending hours on the full run. Use smaller rank/layer coverage and context if memory is tight. Model improvement is a hypothesis, not a completion condition.

## Definition of done
A clean clone can build verified data and run tests without downloading weights; a Mac can reproduce training and all three model runs with documented commands. Measured results and limitations are published, demo starts, CI passes, repository is pushed, and local checkpoints remain usable. No fabricated metrics, hidden oracle demo responses, or test-driven retraining.
