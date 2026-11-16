"""Stockroom: local inference and transparent experiment replay."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
VARIANTS = {"base": "Original model", "fine_tuned": "Fine-tuned · FP16", "quantized": "Fine-tuned · 4-bit"}
EXAMPLES = [
    "Which products are likely to run out in the next 7 days based on sales over the last 14 days?",
    "Which products had the highest return rate in the last 30 days?",
    "What were my top 5 products by net revenue in the last 30 days?",
    "Which products are doing badly?",
]


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def available_stores(data_dir: Path) -> list[str]:
    """Accept manifest versions without hiding stores on malformed metadata."""
    manifest = read_json(data_dir / "manifest.json", {})
    stores = manifest.get("stores", []) if isinstance(manifest, dict) else []
    if isinstance(stores, dict):
        stores = list(stores)
    result = []
    for store in stores:
        if isinstance(store, str):
            result.append(store)
        elif isinstance(store, dict):
            value = store.get("store_id") or store.get("id") or store.get("name")
            if value:
                result.append(str(value))
    if not result:
        result = [path.stem for path in sorted((data_dir / "stores").glob("*.db"))]
    return sorted(set(result))


def read_records(directory: Path) -> list[dict]:
    records = []
    for path in sorted(directory.glob("*.jsonl")):
        try:
            for line in path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    if isinstance(record, dict):
                        record["_record_file"] = path.name
                        records.append(record)
        except (OSError, ValueError):
            continue
    return records


def record_question(record: dict) -> str:
    return str(record.get("question") or record.get("case", {}).get("question") or record.get("case_id") or record.get("id") or "Untitled case")


def summary_frame(summary: Any) -> pd.DataFrame:
    """Render metrics as reported; never invent values for missing measurements."""
    if isinstance(summary, list):
        return pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in summary if isinstance(row, dict)])
    if not isinstance(summary, dict):
        return pd.DataFrame()
    for key in ("runs", "variants", "models", "results"):
        if key in summary:
            return summary_frame(summary[key])
    rows = []
    for variant, values in summary.items():
        if isinstance(values, dict) and any(isinstance(v, (int, float)) for v in values.values()):
            rows.append({"variant": variant, **{k: v for k, v in values.items() if not isinstance(v, (dict, list))}})
    return pd.DataFrame(rows)


def show_result(result: dict, *, replay: bool = False) -> None:
    if replay:
        st.caption("SAVED EXPERIMENT · This is an actual recorded run, not a new inference.")
    if result.get("error"):
        st.error(str(result["error"]))
    action = result.get("action")
    if action == "clarify":
        st.info(result.get("clarification") or result.get("question") or "The model requested clarification.")
    if result.get("explanation"):
        st.markdown(result["explanation"])
        st.caption("Explanation is rendered from returned data; the trained model generates the query or clarification.")
    timing = result.get("timing", {})
    if not timing:
        timing = {key: result[key] for key in ("generation_ms", "total_ms", "peak_memory_bytes", "prompt_tokens", "completion_tokens") if key in result}
    if replay and "correct" in result:
        st.caption("VERIFIED ANSWER: " + ("CORRECT" if result["correct"] else "INCORRECT") + " · QUERY EXECUTED: " + str(result.get("execution_success", "not recorded")))
    if timing:
        with st.expander("Timing and memory"):
            st.json(timing)
    if result.get("sql"):
        st.markdown("**Generated SQL**")
        st.code(result["sql"], language="sql")
    if "rows" in result and result["rows"] is not None:
        st.markdown("**Returned data**")
        rows = result["rows"]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.caption("The query returned no rows.")
    if result.get("assumptions"):
        with st.expander("Assumptions"):
            st.write(result["assumptions"])
    if replay:
        with st.expander("Complete recorded evidence", expanded=not bool(result.get("sql") or result.get("explanation"))):
            st.json({k: v for k, v in result.items() if not k.startswith("_")})


def main() -> None:
    st.set_page_config(page_title="Stockroom · Commerce Task Expert", page_icon="◈", layout="wide")
    st.markdown("""<style>
    html, body, [class*="css"], [class*="st-"] {font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;}
    .stApp {background:#101713;color:#e5ece7;}
    [data-testid="stSidebar"] {background:#151f19;border-right:1px solid #26382c;}
    .block-container {max-width:1260px;padding-top:2.7rem;}
    h1 {font-size:3.5rem!important;letter-spacing:-.065em;font-weight:600!important;line-height:1.05!important;}
    h2,h3 {letter-spacing:-.03em;}
    .eyebrow {color:#96c9a8;font-size:.75rem;letter-spacing:.16em;font-weight:700;margin-bottom:1.1rem;}
    .intro {color:#b0bfb4;font-size:1.06rem;max-width:700px;line-height:1.7;margin:1rem 0 2rem;}
    .badge {display:inline-block;border:1px solid #365642;border-radius:99px;padding:.3rem .7rem;color:#b6d2c0;font-size:.73rem;margin-right:.4rem;}
    div.stButton > button[kind="primary"],div.stFormSubmitButton > button {background:#b6e5c4;color:#102217;border:none;}
    [data-testid="stMetric"] {background:#19271e;border:1px solid #2c4032;border-radius:12px;padding:1rem;}
    [data-testid="stMetricLabel"] {color:#b0bfb4;}
    [data-testid="stMetricValue"] {color:#d4efde;}
    .stTabs [data-baseweb="tab-list"] {gap:2rem;border-bottom:1px solid #293e30;}
    .stTabs [aria-selected="true"] {color:#b6e5c4!important;}
    footer {visibility:hidden;}
    </style>""", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("## ◈ Stockroom")
        st.caption("COMMERCE TASK EXPERT")
        st.divider()
        data_dir = Path(st.text_input("Dataset directory", str(ROOT / "data" / "generated"))).expanduser()
        models_dir = Path(st.text_input("Models directory", str(ROOT / "models"))).expanduser()
        stores = available_stores(data_dir)
        store_id = st.selectbox("Merchant database", stores) if stores else None
        variant = st.selectbox("Model variant", list(VARIANTS), format_func=VARIANTS.get)
        st.divider()
        st.caption("LOCAL BY DESIGN")
        st.write("Synthetic merchants. Read-only queries. Inspectable evidence.")
        st.caption("Model inference and fine-tuning run on your computer. One model is loaded at a time.")
    st.markdown('<div class="eyebrow">SMALL MODEL. REAL EVIDENCE.</div>', unsafe_allow_html=True)
    st.title("Know your store. See the work.")
    st.markdown('<div class="intro">Ask a merchant question and follow the answer from generated SQL to returned data. Compare what specialization improves—and what compression costs.</div>', unsafe_allow_html=True)
    st.markdown('<span class="badge">Apple silicon</span><span class="badge">SQL specialist</span><span class="badge">Verified benchmark</span>', unsafe_allow_html=True)
    st.caption("Synthetic store snapshot · October 1, 2026 · UTC · All question dates use this snapshot.")
    st.write("")
    ask, experiments, methodology = st.tabs(["Ask your store", "Experiment results", "How it works"])
    with ask:
        mode = st.radio("Evidence mode", ["Live inference", "Saved experiment replay"], horizontal=True)
        if mode == "Live inference":
            if not stores:
                st.info("Generate the synthetic stores first: `commerce generate`. No database is connected yet.")
            example = st.selectbox("Try a question", EXAMPLES)
            with st.form("question_form"):
                question = st.text_area("Merchant question", value=example, height=110)
                submitted = st.form_submit_button("Analyze store ↗", type="primary", disabled=not bool(store_id))
            st.caption("Forecasts use a documented recent-sales heuristic. They are estimates, not calibrated probabilities.")
            if submitted:
                with st.spinner("Generating SQL and checking returned data…"):
                    try:
                        from commerce_expert.service import answer_question
                        result = answer_question(question=question, store_id=store_id, variant=variant,
                                                 data_dir=data_dir, models_dir=models_dir)
                        st.session_state["live_result"] = result
                        st.session_state["live_context"] = {"question": question, "store": store_id, "variant": variant}
                    except Exception as exc:  # noqa: BLE001 -- UI boundary displays model/runtime errors to the merchant.
                        st.session_state["live_result"] = {"error": f"{type(exc).__name__}: {exc}"}
            if "live_result" in st.session_state:
                context = st.session_state.get("live_context", {})
                st.caption(f"RESULT · {context.get('store', '')} · {context.get('variant', '')}")
                if context.get("question"):
                    st.markdown(f"**{context['question']}**")
                show_result(st.session_state["live_result"])
        else:
            records = read_records(ROOT / "reports" / "records")
            if not records:
                st.info("No saved experiment records yet. Run the evaluation to inspect real model outputs here.")
            else:
                filenames = sorted({r["_record_file"] for r in records})
                filename = st.selectbox("Experiment run", filenames)
                filtered = [r for r in records if r["_record_file"] == filename]
                index = st.selectbox("Recorded question", range(len(filtered)), format_func=lambda i: record_question(filtered[i]))
                record = filtered[index]
                st.markdown(f"**{record_question(record)}**")
                show_result(record, replay=True)
    with experiments:
        st.subheader("Same questions. Three model variants.")
        st.write("Accuracy and resource usage are measured on the same held-out benchmark. Missing measurements remain missing.")
        summary = read_json(ROOT / "reports" / "summary.json")
        frame = summary_frame(summary)
        if frame.empty:
            st.info("The comparison will appear after the benchmark completes. No results have been invented.")
        else:
            st.dataframe(frame, hide_index=True, width="stretch")
            numeric = [c for c in frame.select_dtypes(include="number").columns if c not in ("count", "n", "total")]
            if numeric:
                metric = st.selectbox("Compare a measured metric", numeric)
                names = frame["variant"] if "variant" in frame else frame.index.astype(str)
                st.bar_chart(pd.DataFrame({"Model": names, metric: frame[metric]}).set_index("Model"), color="#b6e5c4")
        report = ROOT / "reports" / "RESULTS.md"
        if report.exists():
            with st.expander("Read the experiment report"):
                st.markdown(report.read_text())
        if summary is not None:
            with st.expander("Raw measurement summary"):
                st.json(summary)
    with methodology:
        st.subheader("An answer you can audit")
        columns = st.columns(3)
        for column, number, title, text in zip(columns, ["01", "02", "03"],
            ["Generate", "Execute", "Explain"],
            ["The local model receives the question, schema, date, and metric definitions. It generates SQL or requests clarification.",
             "The query is checked and executed against a read-only synthetic store database. Failed queries remain visible.",
             "The response displays the returned rows and documented assumptions, with measurements for the actual run."]):
            with column:
                st.caption(number)
                st.markdown(f"### {title}")
                st.write(text)
        st.divider()
        st.markdown("**What the benchmark checks**")
        st.write("Execution success is only one metric. Query results are compared with independently verified answers. Separate evaluation slices test new merchant data and held-out question patterns. Unclear requests should ask for clarification.")
        manifest = read_json(data_dir / "manifest.json", {})
        if manifest:
            with st.expander("Dataset provenance and policy"):
                st.json(manifest)
        st.caption("This is a portfolio research system using simulated data. Results describe this benchmark and hardware; they do not establish production reliability or general forecasting ability.")


if __name__ == "__main__":
    main()
