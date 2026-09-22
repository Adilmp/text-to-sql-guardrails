"""Command-line interface.

Output goes to stdout as human-readable Rich panels; structured logs go to
``logs/mizan.jsonl``. Keeping those separate means ``mizan ask ... | jq`` works with
``--json`` while the default view stays readable.
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .config import Settings
from .db.build_synthetic import build as build_db
from .errors import MizanError
from .eval import build_suite, injection_suite, run_suite
from .generate import TextToSQL
from .logging import configure, get_logger
from .providers import build_provider
from .schema import load_catalog

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Bilingual Arabic/English Text-to-SQL with AST-level guardrails.",
)
console = Console()
logger = get_logger("cli")

_BAND_COLOUR = {"high": "green", "medium": "yellow", "low": "red"}


def _model_slug(settings: Settings) -> str:
    """Filesystem-safe name of the model that will actually run.

    Every provider must be handled explicitly. An earlier version fell through to
    ``anthropic_model`` for anything that was not Ollama, which wrote **mock** runs into
    ``runs/bilingual-claude-sonnet-5/`` — an artifact directory attributing results to a
    model that never executed. For a project whose central claim is that its numbers are
    measured rather than asserted, a mislabelled run directory is a correctness bug, not a
    naming nit.

    Must match ``scripts/run_eval.py::_slug`` or a run started by one entry point cannot be
    resumed by the other.
    """
    if settings.provider == "ollama":
        model = settings.ollama_model
    elif settings.provider == "anthropic":
        model = settings.anthropic_model
    else:
        model = settings.provider  # "mock"
    return model.replace(":", "-").replace("/", "-")


def _settings(provider: str | None = None, model: str | None = None) -> Settings:
    overrides: dict[str, object] = {}
    if provider:
        overrides["provider"] = provider
    if model and (provider or "ollama") == "ollama":
        overrides["ollama_model"] = model
    elif model and provider == "anthropic":
        overrides["anthropic_model"] = model
    settings = Settings.from_env(**overrides)
    settings.ensure_dirs()
    configure(settings.log_level, settings.log_dir, console=False)
    return settings


@app.command("build-db")
def cmd_build_db(
    orders: int = typer.Option(900, help="Number of orders to generate"),
    customers: int = typer.Option(180, help="Number of customers to generate"),
) -> None:
    """Build the deterministic synthetic demo database."""
    settings = _settings()
    path = build_db(settings.db_path, n_customers=customers, n_orders=orders)
    console.print(f"[green]built[/green] {path}")


@app.command("schema")
def cmd_schema(
    arabic: bool = typer.Option(True, "--arabic/--no-arabic", help="Include Arabic aliases"),
) -> None:
    """Print the schema card exactly as the model sees it."""
    settings = _settings()
    catalog = load_catalog(settings.db_path)
    console.print(Syntax(catalog.to_prompt(include_arabic=arabic), "sql", theme="ansi_dark"))


@app.command("ask")
def cmd_ask(
    question: str = typer.Argument(..., help="Question in Arabic or English"),
    provider: str | None = typer.Option(None, help="ollama | anthropic | mock"),
    model: str | None = typer.Option(None, help="Model name"),
    samples: int = typer.Option(1, "--samples", "-n", help="Self-consistency samples"),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of panels"),
) -> None:
    """Answer one question."""
    settings = _settings(provider, model)
    if samples > 1:
        settings = settings.model_copy(update={"self_consistency_n": samples})

    try:
        catalog = load_catalog(settings.db_path)
        engine = TextToSQL(catalog, build_provider(settings), settings)
        answer = engine.ask(question)
    except MizanError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc

    if as_json:
        console.print_json(json.dumps(answer.to_dict(), ensure_ascii=False))
        raise typer.Exit(0 if answer.ok else 1)

    console.print(Panel(question, title=f"question ({answer.script.value})", border_style="blue"))

    if answer.sql:
        console.print(Syntax(answer.sql, "sql", theme="ansi_dark", word_wrap=True))

    if answer.guardrail and not answer.guardrail.ok:
        table = Table(title="blocked by guardrails", border_style="red")
        table.add_column("rule", style="red")
        table.add_column("detail")
        for violation in answer.guardrail.violations:
            table.add_row(violation.rule, violation.detail)
        console.print(table)
        raise typer.Exit(1)

    if answer.error:
        console.print(f"[red]error:[/red] {answer.error}")
        raise typer.Exit(1)

    if answer.result:
        table = Table(border_style="green")
        for column in answer.result.columns:
            table.add_column(str(column))
        for row in answer.result.rows[:25]:
            table.add_row(*[str(v) for v in row])
        console.print(table)
        if answer.result.truncated:
            console.print("[yellow]result truncated[/yellow]")

    colour = _BAND_COLOUR.get(answer.confidence.band, "white")
    weakest = answer.confidence.weakest
    detail = f" — weakest: {weakest.name} ({weakest.detail})" if weakest else ""
    console.print(
        f"[{colour}]confidence {answer.confidence.score:.2f} "
        f"({answer.confidence.band})[/{colour}]{detail}"
    )
    for warning in answer.guardrail.warnings if answer.guardrail else []:
        console.print(f"[yellow]note:[/yellow] {warning}")
    console.print(f"[dim]{answer.provider}/{answer.model} · {answer.latency_ms:.0f}ms[/dim]")


@app.command("eval")
def cmd_eval(
    provider: str | None = typer.Option(None, help="ollama | anthropic | mock"),
    model: str | None = typer.Option(None, help="Model name"),
    suite: str = typer.Option("bilingual", help="bilingual | injection | both"),
    limit: int | None = typer.Option(None, help="Only run the first N cases"),
    resume: bool = typer.Option(False, "--resume", help="Skip cases already recorded"),
) -> None:
    """Run an evaluation suite and write results to runs/."""
    settings = _settings(provider, model)
    suites = ["bilingual", "injection"] if suite == "both" else [suite]

    for name in suites:
        cases = injection_suite() if name == "injection" else build_suite()
        # `--resume` only means anything if repeated invocations land in the *same* run
        # directory. Without an explicit run_id, `run_suite` mints a fresh timestamped one
        # every time, so resume would read an empty directory, find nothing to skip, and
        # silently redo the entire suite while reporting success. A stable id keyed on
        # suite+model is what makes the flag do what its name says.
        #
        # Without --resume the id stays unique, so a plain re-run still starts clean
        # instead of appending to a previous run's results.
        stable_id = f"{name}-{_model_slug(settings)}" if resume else None
        summary = run_suite(
            settings,
            cases=cases,
            suite_name=name,
            run_id=stable_id,
            resume=resume,
            limit=limit,
        )
        table = Table(title=f"{name} · {summary.provider}/{summary.model}")
        table.add_column("metric")
        table.add_column("value", justify="right")
        table.add_row("cases", str(summary.n))
        table.add_row("correct", str(summary.correct))
        table.add_row("accuracy", f"{summary.accuracy:.1%}")
        table.add_row("blocked", str(summary.blocked))
        table.add_row("mean latency", f"{summary.mean_latency_ms:.0f}ms")
        console.print(table)

        if summary.by_language:
            lang = Table(title="by language", border_style="dim")
            lang.add_column("language")
            lang.add_column("correct/n", justify="right")
            for key, stats in sorted(summary.by_language.items()):
                lang.add_row(key, f"{stats['correct']}/{stats['n']}")
            console.print(lang)


@app.command("health")
def cmd_health(
    provider: str | None = typer.Option(None),
    model: str | None = typer.Option(None),
) -> None:
    """Check that the configured backend and database are usable."""
    settings = _settings(provider, model)
    ok = True

    if settings.db_path.exists():
        console.print(f"[green]ok[/green] database {settings.db_path}")
    else:
        console.print(f"[red]missing[/red] database {settings.db_path} (run: mizan build-db)")
        ok = False

    backend = build_provider(settings)
    if backend.health():
        console.print(f"[green]ok[/green] provider {backend.name}/{backend.model}")
    else:
        console.print(f"[red]unreachable[/red] provider {backend.name}/{backend.model}")
        ok = False

    raise typer.Exit(0 if ok else 1)


@app.command("serve")
def cmd_serve(
    host: str = typer.Option("127.0.0.1", help="Bind address"),
    port: int = typer.Option(8000),
    provider: str | None = typer.Option(None),
    model: str | None = typer.Option(None),
) -> None:
    """Run the local web demo."""
    import os

    import uvicorn

    if provider:
        os.environ["MIZAN_PROVIDER"] = provider
    if model:
        os.environ["MIZAN_OLLAMA_MODEL"] = model

    console.print(f"[green]serving[/green] http://{host}:{port}")
    uvicorn.run("mizan.api:create_app", host=host, port=port, factory=True, log_level="info")


def main() -> None:  # pragma: no cover - console-script shim
    app()


if __name__ == "__main__":  # pragma: no cover
    app()
