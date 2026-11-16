"""Reproducible entry points; ML imports are lazy for CPU-only CI."""

from pathlib import Path

import typer
from rich import print

app = typer.Typer(no_args_is_help=True, help="Local commerce task expert experiments")


@app.command()
def generate(data_dir: Path = Path("data/generated"), seed: int = 42, scale: str = "small"):
    """Build synthetic stores and oracle-verified train/validation/test data."""
    from commerce_expert.benchmark import generate_dataset

    print(generate_dataset(data_dir, seed=seed, scale=scale))


@app.command()
def evaluate(
    variant: str,
    data_dir: Path = Path("data/generated"),
    models_dir: Path = Path("models"),
    reports_dir: Path = Path("reports"),
    limit: int = 0,
    max_tokens: int = 512,
):
    """Evaluate one model in its own process; use identical options for all variants."""
    from commerce_expert.evaluate import run_evaluation

    if variant not in {"base", "fine_tuned", "quantized"}:
        raise typer.BadParameter("Choose base, fine_tuned, or quantized")
    print(run_evaluation(data_dir, models_dir / variant, variant, reports_dir, limit or None, max_tokens))


@app.command()
def ask(
    question: str,
    store: str = "store_01",
    variant: str = "quantized",
    data_dir: Path = Path("data/generated"),
    models_dir: Path = Path("models"),
):
    """Ask a merchant question locally."""
    from commerce_expert.service import answer_question

    print(answer_question(question, store, variant, data_dir, models_dir))


@app.command()
def report(reports_dir: Path = Path("reports")):
    """Merge real run summaries."""
    from commerce_expert.evaluate import merge_reports

    print(merge_reports(reports_dir))


if __name__ == "__main__":
    app()
