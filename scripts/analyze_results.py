"""Build a transparent report from actual per-case records, including paired CIs."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def paired_bootstrap(left: list[dict], right: list[dict], samples: int = 5000) -> dict:
    a = {row["case_id"]: row for row in left}
    b = {row["case_id"]: row for row in right}
    if set(a) != set(b):
        raise ValueError("Paired runs must contain identical case IDs")
    ids = sorted(a)
    diffs = [int(b[k]["correct"]) - int(a[k]["correct"]) for k in ids]
    rng = random.Random(42)
    values = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(samples))
    # These are task-level intervals. Correlated pattern/store instances limit interpretation.
    return {
        "delta": sum(diffs) / len(diffs),
        "ci95": [values[int(0.025 * samples)], values[int(0.975 * samples)]],
        "wins": sum(v > 0 for v in diffs),
        "losses": sum(v < 0 for v in diffs),
        "n": len(ids),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    summary = json.loads((args.reports_dir / "summary.json").read_text())
    records = {}
    for run in summary["runs"]:
        records[run["variant"]] = [
            json.loads(s)
            for s in (args.reports_dir / "records" / f"{run['variant']}.jsonl").read_text().splitlines()
        ]
    paired = {}
    for a, b in [("base", "fine_tuned"), ("fine_tuned", "quantized")]:
        if a in records and b in records:
            paired[f"{a}_to_{b}"] = paired_bootstrap(records[a], records[b])
    (args.reports_dir / "paired.json").write_text(json.dumps(paired, indent=2))
    lines = [
        "# Measured experiment results",
        "",
        "All values below come from actual local MLX runs. No SQL repair or oracle substitution.",
        "",
        "| Variant | Cases | Answer correctness | Query execution | p50 generation | p95 generation | MLX peak |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        lines.append(
            f"| {run['variant']} | {run['n']} | {run['answer_accuracy']:.1%} | {run['query_execution_rate']:.1%} | {run['p50_generation_ms'] / 1000:.2f}s | {run['p95_generation_ms'] / 1000:.2f}s | {run['peak_memory_bytes'] / 1e9:.2f} GB |"
        )
    lines += [
        "",
        "Answer correctness includes correct clarify decisions. Query-only accuracy and individual slices are in summary.json. Clarification wording is not automatically graded. Peak MLX allocations are unified-memory accelerator allocations, not whole-process RAM. Generation time excludes model loading and SQL execution.",
        "",
        "## Generalization slices",
        "",
        "| Variant | Slice | Cases | Correct |",
        "|---|---|---:|---:|",
    ]
    for run in summary["runs"]:
        for name, value in run["slices"].items():
            lines.append(f"| {run['variant']} | {name} | {value['n']} | {value['accuracy']:.1%} |")
    lines += ["", "## Paired comparison", ""]
    for name, value in paired.items():
        lines.append(
            f"- {name}: {value['delta']:+.1%} absolute correctness change; task bootstrap 95% interval [{value['ci95'][0]:+.1%}, {value['ci95'][1]:+.1%}]; {value['wins']} wins, {value['losses']} losses."
        )
    lines += [
        "",
        "Intervals resample cases, which are correlated by store and pattern; they are descriptive and should not be interpreted as independent evidence across hundreds of novel SQL structures.",
        "",
        "## Failure analysis",
        "",
    ]
    for variant, rows in records.items():
        failed = [r for r in rows if not r["correct"]]
        errors = Counter(
            (
                "execution_or_format"
                if r.get("error")
                else "wrong_decision"
                if r.get("action") != r["expected_action"]
                else "wrong_result"
            )
            for r in failed
        )
        lines += [f"### {variant}", "", f"Failures: {dict(errors)}", ""]
        for row in failed[:3]:
            lines += [
                f"- `{row['case_id']}` ({row['pattern_id']}): {row['question']}",
                f"  - Error: {row.get('error') or 'Executable query returned wrong rows or decision.'}",
                f"  - Generated SQL: `{row.get('sql')}`",
            ]
        lines.append("")
    lines += [
        "## Limits",
        "",
        "- Synthetic, shared-schema stores; this evaluates data and composition shifts, not arbitrary schema transfer.",
        "- Small finite pattern library and templated wording limit the breadth of the claim.",
        "- One training seed and one local machine; no universal speed or quality claim.",
        "- Stock risk is constant recent-sales velocity, not a probabilistic forecast; inventory-censored demand is not corrected.",
        "- Exact output aliases are supplied as an evaluation contract; the live demo lets the model choose aliases.",
        "- The explanation is deterministic evidence rendering; the learned component chooses query or clarification.",
        "- Test outputs must not be used to tune the published model or prompt.",
    ]
    (args.reports_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
