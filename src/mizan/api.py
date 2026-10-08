"""FastAPI service for the local web demo.

Two deliberate choices worth defending:

**The catalog and provider are built once, at startup, not per request.** Introspecting the
schema means a handful of PRAGMA queries and constructing the Anthropic client opens a
connection pool; doing either per request would add latency to every call and, for Ollama,
re-check model availability needlessly. They are immutable after construction, so sharing
them across requests is safe.

**Guardrail rejections return HTTP 200, not 4xx.** A blocked query is a *successful*
analysis whose answer is "this was not safe to run". The client needs the violation list to
display it, and burying that in an error status pushes callers toward ``try/except`` around
what is a normal, expected outcome. Genuine failures — a dead backend, a malformed request —
still return real error codes.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .cache import AnswerCache, CachedTextToSQL, CacheSettings
from .clarify import Clarifier
from .config import Settings
from .errors import MizanError, ProviderError
from .generate import TextToSQL
from .logging import configure, get_logger, run_context
from .nl import detect_script
from .present import chart_spec, explain, summarize
from .providers import build_provider
from .questions import Suggester, curated_questions, names_from_db
from .schema import load_catalog

logger = get_logger("api")

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"


class ClarificationChoice(BaseModel):
    #: Which ambiguity the server asked about, and the option the user picked.
    id: str = Field(..., min_length=1, max_length=64)
    option: int = Field(..., ge=0, le=9)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2_000)
    samples: int = Field(default=1, ge=1, le=5)
    #: The id of the answer this question follows up on. The server looks that answer up
    #: itself; the browser never supplies the earlier SQL.
    follow_up_of: str | None = Field(default=None, min_length=8, max_length=64)
    #: The user's answer to a clarifying question the server asked about this question.
    clarification: ClarificationChoice | None = None


class FeedbackRequest(BaseModel):
    #: The id the server gave the answer. The SQL itself is never accepted from the client:
    #: a verdict can only be about a query this server produced and validated.
    answer_id: str = Field(..., min_length=8, max_length=64)
    verdict: Literal["up", "down"]


class AskResponse(BaseModel):
    ok: bool
    question: str
    script: str
    rtl: bool
    sql: str | None
    result: dict[str, Any] | None
    confidence: dict[str, Any]
    violations: list[dict[str, str]]
    warnings: list[str]
    error: str | None
    provider: str
    model: str
    latency_ms: float
    #: How many times the model was asked to fix its own query before this answer.
    repairs: int = 0
    #: Set when the answer came from the cache: which earlier question it matched.
    cache: dict[str, Any] | None = None
    #: The answer as one line in the question's language, built from the result (present.py).
    summary: str | None = None
    #: Which chart fits the result: {"type": "bar" | "line", "label": col, "value": col}.
    chart: dict[str, Any] | None = None
    #: The answer as plain steps in the question's language (present.explain).
    explanation: str | None = None
    #: Pass back to /api/feedback to say whether the answer was right.
    answer_id: str | None = None
    #: Someone already reported exactly this query as the wrong answer to this question.
    reported: bool = False
    #: Set instead of an answer when the question needs a choice first (clarify.py):
    #: {"id", "ask", "options": [labels]}, in the question's language.
    clarify: dict[str, Any] | None = None
    #: The meaning the user chose, when the question was clarified.
    clarified_as: str | None = None
    #: The earlier question this one followed up on.
    follow_up_of: str | None = None


#: How often an idle server re-sends its warm-up. Well inside the 30-minute keep-alive, so the
#: model never unloads while the server is up.
KEEP_WARM_EVERY_S = 600.0


def create_app(
    settings: Settings | None = None, cache_settings: CacheSettings | None = None
) -> FastAPI:
    """Application factory.

    A factory rather than a module-level ``app`` so tests can build an instance with
    injected settings (a temporary database, the mock provider) without touching the
    environment.
    """
    cfg = settings or Settings.from_env()
    if cache_settings is None:
        # Injected settings mean a test or an embedding app: no cache file unless asked for.
        cache_settings = CacheSettings.from_env() if settings is None else CacheSettings(False)
    cfg.ensure_dirs()
    configure(cfg.log_level, cfg.log_dir, console=False)

    state: dict[str, Any] = {"warmup": "running", "last_used": time.monotonic()}
    #: Recent answers by id, so feedback names an answer instead of carrying SQL. Bounded:
    #: feedback is given right after an answer, not days later.
    recent: OrderedDict[str, tuple[str, str, str]] = OrderedDict()
    recent_lock = threading.Lock()
    ready = threading.Event()
    stop_keep_warm = threading.Event()

    def keep_warm() -> None:
        """While the server runs, keep the model loaded and its prompt processed.

        Ollama unloads an idle model after its keep-alive (30 minutes here), and the next
        visitor then pays the whole cold start: minutes on a CPU. Re-sending the normal
        warm-up when nobody has asked anything for a while resets that timer and costs a
        second or two, because the prompt is still cached. It goes through the same provider
        call as every question on purpose: an earlier version sent Ollama a bare keep-alive
        request instead, without the context size the model was loaded with, and Ollama
        *reloaded* the model, throwing the processed prompt away (DECISIONS.md D40). When the
        server stops, so do these, and Ollama unloads the model on its own.
        """
        while not stop_keep_warm.wait(KEEP_WARM_EVERY_S):
            if time.monotonic() - state["last_used"] >= KEEP_WARM_EVERY_S:
                state["engine"].warm_up()

    def warm_up() -> None:
        try:
            ok = state["engine"].warm_up() is not None
            state["warmup"] = "ready" if ok else "failed"
        except Exception:  # never let the warm-up thread die silently with "running"
            state["warmup"] = "failed"
            raise
        finally:
            ready.set()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        state["catalog"] = load_catalog(cfg.db_path)
        state["provider"] = build_provider(cfg)
        state["engine"] = TextToSQL(state["catalog"], state["provider"], cfg)
        cache = AnswerCache(cache_settings.path) if cache_settings.enabled else None
        state["cache"] = cache
        state["served"] = CachedTextToSQL(state["engine"], cache, cache_settings)
        # Ambiguous business terms, next to the glossary: <db>.clarifications.json.
        state["clarifier"] = Clarifier.from_file(cfg.db_path.with_suffix(".clarifications.json"))
        state["suggester"] = Suggester(q for _, q in curated_questions(names_from_db(cfg.db_path)))
        if state["provider"].name != "mock":
            # Read the prompt into the model's cache now, in the background, so the first
            # visitor doesn't wait minutes on a CPU. A request that arrives first queues
            # behind it in the model server; /api/health reports `warming_up` meanwhile.
            threading.Thread(target=warm_up, daemon=True).start()
            threading.Thread(target=keep_warm, daemon=True).start()
        else:
            state["warmup"] = "ready"
            ready.set()
        logger.info(
            "api ready",
            extra={"provider": state["provider"].name, "model": state["provider"].model},
        )
        yield
        stop_keep_warm.set()
        if cache is not None:
            cache.close()
        state.clear()

    app = FastAPI(title="Text-to-SQL Guardrails", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Response:
        """Response headers that bound the damage if the escaping above ever fails.

        ``default-src 'none'`` plus ``connect-src 'self'`` is the load-bearing pair: it
        means injected script cannot reach an external host, which removes the payoff from
        most XSS even if a payload executes. ``frame-ancestors`` and ``X-Frame-Options``
        block clickjacking of a locally-bound tool — relevant because a page in the user's
        own browser can reach ``localhost``.

        ``'unsafe-inline'`` is present because the demo is a deliberately self-contained
        single file. That is a real weakening and is recorded as such in SECURITY.md rather
        than glossed over; splitting the script and style into separate files and dropping
        to ``'self'`` is the upgrade path.

        No CORS middleware is installed anywhere, which is itself the CORS policy: requests
        are same-origin only. Combined with FastAPI requiring a JSON content type (which
        forces a preflight that no cross-origin caller can satisfy), this is also what
        stands in for CSRF protection on a tool with no cookies and no ambient authority.
        """
        response: Response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; "
            "script-src 'unsafe-inline'; "
            "style-src 'unsafe-inline'; "
            "connect-src 'self'; "
            "base-uri 'none'; "
            "form-action 'none'; "
            "frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        page = WEB_DIR / "index.html"
        if not page.exists():
            raise HTTPException(status_code=500, detail="web/index.html is missing")
        return page.read_text(encoding="utf-8")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        provider = state.get("provider")
        served: CachedTextToSQL | None = state.get("served")
        cache = state.get("cache")
        return {
            "warming_up": not ready.is_set(),
            # "ready", "running" or "failed": a warm-up that failed is not readiness.
            "warmup": state.get("warmup"),
            "cached_answers": cache.count(served.context) if cache and served else None,
            "ok": cfg.db_path.exists() and provider is not None,
            "database": str(cfg.db_path),
            "provider": provider.name if provider else None,
            "model": provider.model if provider else None,
        }

    @app.post("/api/feedback")
    def feedback(request: FeedbackRequest) -> dict[str, Any]:
        """👍 marks an answer verified; 👎 stops it being served from the cache and flags it
        for the next person. Every verdict is logged for review (logs/feedback.jsonl)."""
        with recent_lock:
            entry = recent.get(request.answer_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="unknown or expired answer id")
        question, sql, _ = entry
        served: CachedTextToSQL = state["served"]
        up = request.verdict == "up"
        served.verdict(question, sql, up)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "verdict": request.verdict,
            "question": question,
            "sql": sql,
            "model": state["provider"].model,
        }
        with (cfg.log_dir / "feedback.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"ok": True, "status": "verified" if up else "reported"}

    @app.get("/api/suggest")
    async def suggest(
        q: str = Query("", max_length=300), limit: int = Query(8, ge=1, le=20)
    ) -> dict[str, Any]:
        """Curated questions matching what has been typed, cached ("instant") ones first."""
        served: CachedTextToSQL = state["served"]
        cache = state.get("cache")

        def status_of(key: str) -> int | None:
            entry = cache.status(served.context, key) if cache is not None else None
            return entry[1] if entry else None

        found = state["suggester"].suggest(q, status_of=status_of, limit=limit)
        return {"suggestions": [s.to_dict() for s in found]}

    @app.get("/api/schema")
    async def schema() -> dict[str, Any]:
        catalog = state["catalog"]
        return {
            "ddl": catalog.to_prompt(languages=("ar", "ur")),
            "tables": {
                name: {
                    "columns": [c.name for c in table.columns],
                    "description": table.description,
                    "aliases_ar": list(table.aliases_ar),
                    "aliases_ur": list(table.aliases_ur),
                    "row_count": table.row_count,
                }
                for name, table in sorted(catalog.tables.items())
            },
        }

    # A plain `def`, not `async def`: answering blocks on the model for seconds to minutes.
    # FastAPI runs sync handlers in a thread pool; an async handler doing the same blocking
    # call would freeze the event loop, and with it every other request, health checks
    # included, until the model replied.
    @app.post("/api/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        served: CachedTextToSQL = state["served"]
        clarifier: Clarifier = state["clarifier"]
        state["last_used"] = time.monotonic()
        hit = None
        question, clarified_as = request.question, None

        if request.clarification is not None:
            try:
                question, clarified_as = clarifier.apply(
                    question, request.clarification.id, request.clarification.option
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        elif (ambiguity := clarifier.check(question)) is not None:
            # Ask before answering: no model call, so this costs nothing.
            script = detect_script(question)
            return AskResponse(
                ok=False,
                question=question,
                script=script.value,
                rtl=script.is_rtl,
                sql=None,
                result=None,
                confidence={"score": 0.0, "band": "low", "signals": []},
                violations=[],
                warnings=[],
                error=None,
                provider=state["provider"].name,
                model=state["provider"].model,
                latency_ms=0.0,
                clarify=ambiguity.to_dict(script.prompt_language),
            )

        context: tuple[tuple[str, str], ...] = ()
        follow_up_of = None
        if request.follow_up_of:
            with recent_lock:
                earlier = recent.get(request.follow_up_of)
            if earlier is not None:  # an expired id: answer the question on its own
                follow_up_of, _, earlier_reply = earlier
                context = ((follow_up_of, earlier_reply),)

        with run_context() as run_id:
            try:
                if request.samples > 1:
                    # Several samples are a request for the model's own agreement, so they
                    # bypass the cache.
                    engine = TextToSQL(
                        state["catalog"],
                        state["provider"],
                        cfg.model_copy(update={"self_consistency_n": request.samples}),
                    )
                    answer = engine.ask(question, context)
                else:
                    answer, hit = served.ask(question, context)
            except ProviderError as exc:
                logger.error("provider failure", extra={"error": str(exc), "run_id": run_id})
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except MizanError as exc:  # pragma: no cover - defensive
                logger.error("unexpected failure", extra={"error": str(exc)})
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        guardrail = answer.guardrail
        answer_id = None
        if answer.ok and answer.sql:
            answer_id = uuid.uuid4().hex
            with recent_lock:
                # The model's own reply where there is one: a follow-up puts it back in the
                # conversation, and identical text lets the model server reuse its cache.
                recent[answer_id] = (answer.question, answer.sql, answer.raw_output or answer.sql)
                while len(recent) > 500:
                    recent.popitem(last=False)
        return AskResponse(
            ok=answer.ok,
            question=answer.question,
            script=answer.script.value,
            rtl=answer.script.is_rtl,
            sql=answer.sql,
            result=answer.result.to_dict() if answer.result else None,
            confidence=answer.confidence.to_dict(),
            violations=(
                [{"rule": v.rule, "detail": v.detail} for v in guardrail.violations]
                if guardrail
                else []
            ),
            warnings=list(guardrail.warnings) if guardrail else [],
            error=answer.error,
            provider=answer.provider,
            model=answer.model,
            latency_ms=round(answer.latency_ms, 1),
            repairs=answer.repairs,
            cache=hit.to_dict() if hit else None,
            summary=summarize(answer.question, answer.script, answer.result)
            if answer.result
            else None,
            chart=chart_spec(answer.result) if answer.result else None,
            explanation=explain(answer.sql, state["catalog"], answer.script)
            if answer.ok and answer.sql
            else None,
            answer_id=answer_id,
            reported=bool(answer.sql)
            and hit is None
            and served.reported(question, answer.sql or ""),
            clarified_as=clarified_as,
            follow_up_of=follow_up_of,
        )

    return app
