# Stockroom demo

Run from the repository root after installing dependencies and generating data:

```sh
uv run streamlit run app.py
```

Open the local URL printed by Streamlit. The app is local, and no hosted model or paid API is used.

## Live inference

Choose a generated merchant database, then Original, Fine-tuned FP16, or Fine-tuned 4-bit. Ask a question and inspect the generated SQL, database rows, explanation, and timing. Model files are expected under `models/base`, `models/fine_tuned`, and `models/quantized`; the model service owns loading and releases the previous model when switching variants. The UI never substitutes reference SQL for a model prediction.

If local artifacts have not been generated, the app shows setup guidance rather than sample answers. Missing models and runtime failures remain visible. Ambiguous requests should produce a clarification instead of SQL.

## Saved experiment replay

The replay tab reads actual evaluation records from `reports/records/*.jsonl`. Replay is explicitly labeled and does not claim to perform new inference. Complete recorded evidence remains available even when a record has no standard display fields.

The results tab reads `reports/summary.json` and optional `reports/RESULTS.md`. It only displays reported metrics, with no hard-coded accuracy or performance figures. Use replay to present the experiment without loading a model on stage.

## Demo narrative

1. Show a stockout-risk question and inspect its assumptions and returned data.
2. Switch to a return-rate question that requires joining orders and returns.
3. Show an ambiguous question and the model's actual behavior, including failures.
4. Compare the three variants in experiment results.
5. Open a failed held-out case and explain its error category.

The inventory estimate uses a documented recent-sales heuristic. It is not a probabilistic demand forecasting model. Store databases share a schema, so held-out stores test data distribution generalization rather than arbitrary schema transfer.
