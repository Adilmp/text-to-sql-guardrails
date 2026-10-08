"""Answer cache: a repeated question is answered in milliseconds instead of seconds.

What is cached, and why not the rows
------------------------------------
The **validated SQL**, not the result. A cache hit re-runs the query, which takes
milliseconds on this database, so an answer is never staler than the data it reads (DDIA
ch. 1 and 11: the cache is derived data, and derived data must not drift from its source).
What the model costs, seconds of CPU per question, is the part worth skipping.

What counts as "the same question"
----------------------------------
Two questions share an entry only when they differ in ways that cannot change the answer:
case, spacing, punctuation, polite filler ("please", "من فضلك", "براہ کرم"), Arabic and Urdu
spelling and keyboard variants, and which digits were typed (:func:`canonical_question`).

Paraphrases ("how many late deliveries were there?") are deliberately **not** matched. That
was measured before it was ruled out (DECISIONS.md D40): the local embedding model scores
"customers in Dubai" against "customers in Riyadh" at 1.000 and "late most often" against
"late least often" at 0.995, higher than true rewordings (0.94–0.98), and two different
Arabic questions at 0.988. A similarity cache would serve confident wrong answers, which is
worse than no cache.

What is never cached
--------------------
Only answers the pipeline was sure of go in: executed, high confidence, first attempt (no
repair), every filter value found in the data. A blocked or repaired answer would make a
one-off model mistake sticky for every later user.

The cache file is untrusted
---------------------------
It is a file on disk, and anything that can write to it can plant SQL. Every hit is
validated again by the same AST guardrails as a fresh model answer before it runs; one that
fails is deleted and the question goes to the model. The cache can make an answer faster,
never less checked.

Invalidation
------------
Entries are keyed by a *context*: a hash of the system prompt (which contains the schema,
value lists and row counts), the backend and the model. Change any of them and earlier
entries simply stop matching. Nothing has to remember to clear the cache.

Settings live here, not in ``config.py``: ``config.py`` holds what can change an answer, and
that boundary is what the behaviour fingerprint draws (D28). A cache changes how fast an
answer arrives, not what it is.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import PROJECT_ROOT
from .errors import MizanError
from .generate.pipeline import Answer, TextToSQL
from .guardrails import execute, validate
from .logging import get_logger
from .nl import clean_for_model, detect_script, normalize_for_matching
from .validate.confidence import HIGH_THRESHOLD, score_answer

logger = get_logger("cache")

#: Words that ask politely or point at the answer and never change it. Deliberately short:
#: anything that could carry meaning (a negation, a comparison, a verb, a noun) stays.
_FILLER_WORDS = frozenset(
    {
        # English
        "please",
        "kindly",
        "the",
        "a",
        "an",
        # Arabic ("please", "if you allow"), already letter-folded by normalize_for_matching
        "رجاء",
        "رجاءا",
        # Urdu ("please", "kindly", "just"), letter-folded
        "پليز",
        "ذرا",
    }
)
#: Multi-word fillers, matched as whole phrases after folding.
_FILLER_PHRASES = (
    "can you tell me",
    "could you tell me",
    "tell me",
    "من فضلك",
    "لو سمحت",
    "براه كرم",
    "مهرباني كر كي",
    "مهرباني سي",
)

_NON_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)


def canonical_question(text: str) -> str:
    """The cache key: the question with only meaning-free variation removed.

    Folds what :func:`normalize_for_matching` folds (case, diacritics, Arabic and Urdu letter
    variants, digit scripts, invisible characters), then drops punctuation and the filler
    words above. Word order and every other word are kept, so "not", "late", "most", a city
    or a number always distinguish two questions.
    """
    folded = normalize_for_matching(text)
    folded = _NON_WORD_RE.sub(" ", folded).strip()
    padded = f" {folded} "
    for phrase in _FILLER_PHRASES:
        padded = padded.replace(f" {normalize_for_matching(phrase)} ", " ")
    return " ".join(w for w in padded.split() if w not in _FILLER_WORDS)


@dataclass(frozen=True)
class CacheSettings:
    """How the server caches answers. Read from ``MIZAN_CACHE*`` environment variables."""

    enabled: bool = True
    path: Path = PROJECT_ROOT / ".cache" / "answers.sqlite"
    #: Answers below this confidence are not cached (D17's "high" band).
    min_confidence: float = HIGH_THRESHOLD

    @classmethod
    def from_env(cls) -> CacheSettings:
        flag = os.environ.get("MIZAN_CACHE", "on").strip().lower()
        path = os.environ.get("MIZAN_CACHE_PATH")
        return cls(
            enabled=flag not in ("off", "0", "false", "no"),
            path=Path(path) if path else cls.path,
        )


@dataclass(frozen=True)
class CacheHit:
    """Where a cached answer came from, for the response and the UI."""

    question: str
    hits: int
    created_at: str

    def to_dict(self) -> dict[str, object]:
        return {"hit": True, "matched": self.question, "hits": self.hits, "since": self.created_at}


def cache_context(engine: TextToSQL) -> str:
    """Everything a cached query depends on. Change any of it and old entries stop matching."""
    digest = hashlib.sha256()
    for part in (
        engine.system_prompt,
        engine.provider.name,
        engine.provider.model,
        str(engine.settings.temperature),
    ):
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


class AnswerCache:
    """A small SQLite table of validated queries, keyed by context and canonical question.

    Safe to share between the API's worker threads: one connection guarded by a lock.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS answers ("
                " context TEXT NOT NULL, key TEXT NOT NULL, question TEXT NOT NULL,"
                " sql TEXT NOT NULL, confidence REAL NOT NULL, created_at TEXT NOT NULL,"
                " hits INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (context, key))"
            )
            self._conn.commit()

    def get(self, context: str, key: str) -> tuple[str, str, int, str] | None:
        """``(question, sql, hits, created_at)`` for a stored entry, and count the hit."""
        with self._lock:
            row = self._conn.execute(
                "SELECT question, sql, hits, created_at FROM answers WHERE context = ? AND key = ?",
                (context, key),
            ).fetchone()
            if row is None:
                return None
            self._conn.execute(
                "UPDATE answers SET hits = hits + 1 WHERE context = ? AND key = ?", (context, key)
            )
            self._conn.commit()
        return str(row[0]), str(row[1]), int(row[2]) + 1, str(row[3])

    def put(self, context: str, key: str, question: str, sql: str, confidence: float) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO answers (context, key, question, sql, confidence,"
                " created_at, hits) VALUES (?, ?, ?, ?, ?, ?, 0)",
                (context, key, question, sql, confidence, now),
            )
            self._conn.commit()

    def delete(self, context: str, key: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM answers WHERE context = ? AND key = ?", (context, key))
            self._conn.commit()

    def count(self, context: str | None = None) -> int:
        with self._lock:
            if context is None:
                row = self._conn.execute("SELECT COUNT(*) FROM answers").fetchone()
            else:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM answers WHERE context = ?", (context,)
                ).fetchone()
        return int(row[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def cacheable(answer: Answer, min_confidence: float = HIGH_THRESHOLD) -> bool:
    """Whether the pipeline was sure enough of this answer to reuse it for everyone."""
    if not (answer.ok and answer.sql and answer.guardrail and answer.guardrail.ok):
        return False
    if answer.repairs or answer.confidence.score < min_confidence:
        return False
    signals = {s.name: s.value for s in answer.confidence.signals}
    return signals.get("values_grounded", 0.0) == 1.0 and signals.get("first_attempt", 0.0) == 1.0


class CachedTextToSQL:
    """The serving front of the pipeline: answer from the cache when it is safe, else ask.

    The eval harness never goes through this class, so measured accuracy and latency are
    always the model's own.
    """

    def __init__(self, engine: TextToSQL, cache: AnswerCache | None, settings: CacheSettings):
        self.engine = engine
        self.cache = cache if settings.enabled else None
        self.settings = settings
        self.context = cache_context(engine)

    def ask(self, question: str) -> tuple[Answer, CacheHit | None]:
        key = canonical_question(question)
        if self.cache is None or not key:
            return self.engine.ask(question), None
        if (served := self._from_cache(question, key)) is not None:
            return served
        answer = self.engine.ask(question)
        if answer.sql is not None and cacheable(answer, self.settings.min_confidence):
            self.cache.put(self.context, key, question, answer.sql, answer.confidence.score)
        return answer, None

    def _from_cache(self, question: str, key: str) -> tuple[Answer, CacheHit] | None:
        if self.cache is None:  # pragma: no cover - guarded by the caller
            return None
        started = time.perf_counter()
        entry = self.cache.get(self.context, key)
        if entry is None:
            return None
        cached_question, sql, hits, created_at = entry

        # Validated again, exactly like a fresh model answer: the cache file is untrusted.
        report = validate(sql, self.engine.catalog, self.engine.policy)
        if not report.ok:
            logger.warning(
                "cached query failed validation; evicted",
                extra={"rules": report.rules_fired, "question": cached_question[:80]},
            )
            self.cache.delete(self.context, key)
            return None
        try:
            result = execute(
                report.sql,
                self.engine.settings.db_path,
                max_rows=self.engine.settings.max_rows,
                timeout_s=self.engine.settings.query_timeout_s,
            )
        except MizanError as exc:
            logger.warning("cached query failed to run; evicted", extra={"error": str(exc)})
            self.cache.delete(self.context, key)
            return None

        confidence = score_answer(
            schema_grounded=True,
            executed=True,
            row_count=result.row_count,
            agreement=None,
            rewrite_warnings=len(report.warnings),
        )
        answer = Answer(
            question=question,
            question_clean=clean_for_model(question),
            script=detect_script(question),
            sql=report.sql,
            guardrail=report,
            result=result,
            confidence=confidence,
            provider=self.engine.provider.name,
            model=self.engine.provider.model,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        logger.info("answered from cache", extra={"hits": hits})
        return answer, CacheHit(cached_question, hits, created_at)
