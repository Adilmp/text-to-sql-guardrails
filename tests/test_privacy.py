"""The leak test: no stored private value reaches the model, on any path.

test_placeholders.py follows one name through each stage. This file sweeps: **every**
customer name in the test database, in English, Arabic and Urdu questions, through every
way the pipeline talks to the model (first answer, each kind of repair, self-consistency
samples, follow-ups), and checks everything the model was shown for **any** customer's
name, not only the one asked about. A repair hint that suggested "did you mean Tariq Al
Otaibi?" would leak a name nobody typed, and only a sweep like this one catches it.

Names are compared after ``normalize_for_matching`` as well as verbatim, so a leak in a
different spelling (no madda, Urdu letters, lower case) still counts. The engine's log
records are checked the same way: a log line is a copy of the data too.

This is the measurement behind the claim "0 private values sent to the model"
(DECISIONS.md D44); the eval reports the same count on real model output (task 11).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path

import pytest

from mizan.config import Settings
from mizan.generate import TextToSQL
from mizan.nl import normalize_for_matching
from mizan.privacy import Masker
from mizan.providers import MockProvider
from mizan.schema import Catalog

SENSITIVE = {"customers.name_en": "customer_name", "customers.name_ar": "customer_name_ar"}

JOIN = "FROM orders o JOIN customers c ON c.customer_id = o.customer_id"
GOOD_EN = f"SELECT COUNT(*) {JOIN} WHERE c.name_en = :customer_name_1"
GOOD_AR = f"SELECT COUNT(*) {JOIN} WHERE c.name_ar = :customer_name_ar_1"

#: (question template, the reply that answers it). Urdu questions name the customer in
#: Arabic script, the way an Urdu speaker would write an Arab name.
TEMPLATES = [
    ("how many orders did {en} place?", GOOD_EN),
    ("what did {en} spend in total, and when was the last order?", GOOD_EN),
    ("كم طلب قدم {ar}؟", GOOD_AR),  # "How many orders did ... place?"
    ("{ar} نے کتنے آرڈر دیے؟", GOOD_AR),  # Urdu: "How many orders did ... place?"
]

#: First replies that each send the pipeline down a different repair path, then the fix.
REPAIR_PATHS = {
    "quoted_placeholder": f"SELECT COUNT(*) {JOIN} WHERE c.name_en = ':customer_name_1'",
    "unknown_placeholder": f"SELECT COUNT(*) {JOIN} WHERE c.name_en = :customer_name_9",
    "placeholder_on_wrong_column": f"SELECT COUNT(*) {JOIN} WHERE c.city = :customer_name_1",
    "unknown_column": f"SELECT COUNT(*) {JOIN} WHERE c.full_name = :customer_name_1",
    "ambiguous_column": (
        f"SELECT customer_id, COUNT(*) {JOIN} WHERE c.name_en = :customer_name_1 "
        "GROUP BY customer_id"
    ),
    "parse_error": "SELECT COUNT(* FROM orders WHERE name_en = :customer_name_1",
}


@pytest.fixture(scope="module")
def masker(db_path: Path) -> Masker:
    return Masker.from_db(db_path, SENSITIVE)


@pytest.fixture(scope="module")
def customers(db_path: Path) -> list[tuple[str, str]]:
    """Every distinct (English name, Arabic name) pair in the test database."""
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        return list(conn.execute("SELECT DISTINCT name_en, name_ar FROM customers ORDER BY 1"))


@pytest.fixture(scope="module")
def all_names(customers: list[tuple[str, str]]) -> list[str]:
    return [name for pair in customers for name in pair]


class _Capture(logging.Handler):
    """Every record the engine logs, with its structured extras."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def text(self) -> str:
        return "\n".join(
            json.dumps(r.__dict__, default=str, ensure_ascii=False) for r in self.records
        )


@pytest.fixture()
def logs() -> Iterator[_Capture]:
    capture = _Capture()
    root = logging.getLogger("mizan")
    level = root.level
    root.addHandler(capture)
    root.setLevel(logging.DEBUG)
    try:
        yield capture
    finally:
        root.removeHandler(capture)
        root.setLevel(level)


def _sent(provider: MockProvider) -> str:
    """Everything the model was shown, across every call: system, history and new turn."""
    return "\n".join(
        "\n".join([call.system, call.user, *(text for _, text in call.messages)])
        for call in provider.calls
    )


def _leaks(text: str, names: list[str]) -> list[str]:
    folded = normalize_for_matching(text)
    return [n for n in names if n in text or normalize_for_matching(n) in folded]


def _engine(
    catalog: Catalog, settings: Settings, masker: Masker, *script: str
) -> tuple[TextToSQL, MockProvider]:
    provider = MockProvider(script=list(script))
    return TextToSQL(catalog, provider, settings, masker=masker), provider


def test_the_system_prompt_holds_no_customer_name(
    catalog: Catalog, settings: Settings, masker: Masker, all_names: list[str]
) -> None:
    # Sample values are only taken from short categorical columns; a name column must
    # never be one of them, or every request would carry customer names.
    engine, _ = _engine(catalog, settings, masker)
    assert _leaks(engine.system_prompt, all_names) == []


@pytest.mark.parametrize(("template", "reply"), TEMPLATES)
def test_no_customer_name_reaches_the_model(
    catalog: Catalog,
    settings: Settings,
    masker: Masker,
    customers: list[tuple[str, str]],
    all_names: list[str],
    logs: _Capture,
    template: str,
    reply: str,
) -> None:
    for en, ar in customers:
        engine, provider = _engine(catalog, settings, masker, reply)
        answer = engine.ask(template.format(en=en, ar=ar))
        assert answer.ok, (en, answer.error)
        assert _leaks(_sent(provider), all_names) == [], en
    assert _leaks(logs.text(), all_names) == []


@pytest.mark.parametrize("path", sorted(REPAIR_PATHS))
def test_no_customer_name_reaches_the_model_during_repair(
    catalog: Catalog,
    settings: Settings,
    masker: Masker,
    customers: list[tuple[str, str]],
    all_names: list[str],
    logs: _Capture,
    path: str,
) -> None:
    for en, _ in customers:
        engine, provider = _engine(catalog, settings, masker, REPAIR_PATHS[path], GOOD_EN)
        answer = engine.ask(f"how many orders did {en} place?")
        assert answer.repairs == 1, (path, en)
        assert answer.ok, (path, en, answer.error)
        assert _leaks(_sent(provider), all_names) == [], (path, en)
    assert _leaks(logs.text(), all_names) == []


def _misspell(name: str) -> str:
    """Swap two letters in the middle of the last word: "Al Thani" -> "Al Thnai"."""
    *rest, last = name.split()
    i = len(last) // 2
    return " ".join([*rest, last[: i - 1] + last[i] + last[i - 1] + last[i + 1 :]])


def test_a_misspelled_name_is_not_corrected_into_a_stored_one(
    catalog: Catalog,
    settings: Settings,
    masker: Masker,
    customers: list[tuple[str, str]],
    all_names: list[str],
    logs: _Capture,
) -> None:
    # A misspelled name isn't a stored value, so it isn't masked (D44's known limit) and the
    # model may filter on it as a literal. Grounding would normally answer "did you mean
    # 'Tariq Al Thani'?", which hands the model the real, stored name. It must not.
    checked = 0
    for en, _ in customers:
        typo = _misspell(en)
        if typo == en or _leaks(typo, all_names):
            continue
        engine, provider = _engine(
            catalog,
            settings,
            masker,
            f"SELECT COUNT(*) {JOIN} WHERE c.name_en = '{typo}'",
            f"SELECT COUNT(*) {JOIN} WHERE c.name_en = '{typo}'",
        )
        engine.ask(f"how many orders did {typo} place?")
        assert _leaks(_sent(provider), all_names) == [], typo
        checked += 1
    assert checked >= len(customers) // 2
    assert _leaks(logs.text(), all_names) == []


def test_no_customer_name_reaches_the_model_across_samples(
    catalog: Catalog,
    settings: Settings,
    masker: Masker,
    customers: list[tuple[str, str]],
    all_names: list[str],
) -> None:
    sampled = settings.model_copy(update={"self_consistency_n": 3})
    en, _ = customers[0]
    provider = MockProvider(script=[GOOD_EN] * 3)
    engine = TextToSQL(catalog, provider, sampled, masker=masker)
    answer = engine.ask(f"how many orders did {en} place?")
    assert answer.ok and len(provider.calls) == 3
    assert _leaks(_sent(provider), all_names) == []


def test_no_customer_name_reaches_the_model_in_a_follow_up(
    catalog: Catalog,
    settings: Settings,
    masker: Masker,
    customers: list[tuple[str, str]],
    all_names: list[str],
    logs: _Capture,
) -> None:
    both = f"SELECT COUNT(*) {JOIN} WHERE c.name_en IN (:customer_name_1, :customer_name_2)"
    for (first, _), (second, _) in pairwise(customers):
        engine, provider = _engine(catalog, settings, masker, both)
        answer = engine.ask(
            f"and {second}?",
            context=[(f"how many orders did {first} place?", GOOD_EN)],
        )
        assert answer.ok, (first, second, answer.error)
        assert _leaks(_sent(provider), all_names) == [], (first, second)
    assert _leaks(logs.text(), all_names) == []


@pytest.mark.parametrize(
    "value",
    ["tariq@example.com", "+92 300 1234567", "+971 50 123 4567", "42101-1234567-1"],
)
def test_contact_details_never_reach_the_model(
    catalog: Catalog, settings: Settings, masker: Masker, logs: _Capture, value: str
) -> None:
    engine, provider = _engine(catalog, settings, masker, "SELECT COUNT(*) FROM customers")
    engine.ask(f"which customer has {value}?")
    assert value not in _sent(provider)
    assert value not in logs.text()
