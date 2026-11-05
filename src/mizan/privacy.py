"""Private values in a question become placeholders before the question reaches the model.

"How many orders did Tariq Al Thani place?" is sent to the model as "How many orders did
:customer_name_1 place?", together with one line saying that ``:customer_name_1`` is a value
of ``customers.name_en``. The model writes ``WHERE c.name_en = :customer_name_1``, and the
real value is bound as a SQLite parameter when the query runs. It never appears in a prompt,
in the generated SQL text, in the answer cache or in the logs (DECISIONS.md D44).

What counts as private
----------------------
* **Stored values of the columns listed in** ``<db>.privacy.json``: customer names, in
  English and in Arabic script. Matching a known list is deterministic and cannot miss a
  stored value, which a name-recognition model can, least of all in Urdu. Matching runs on
  ``normalize_for_matching`` keys, so a name typed with an Urdu keyboard (ی for ي) or without
  the madda (ال for آل) is still found, and the binding carries the stored spelling, so the
  SQL comparison matches.
* **Emails, international phone numbers (+92, +96x, +97x) and CNIC / Emirates ID numbers**,
  by shape. They are masked even though no column holds them: a value the database cannot
  use is still not the model's to see.

Only whole stored values are matched, on word boundaries. A first name alone ("Tariq") is
not, because a single common word matches far more than the person meant; that limit is
recorded in D44 rather than papered over with fuzzy matching.

A question with nothing private in it is returned exactly as given, which is what keeps the
eval suites byte-identical with masking switched on (tests/test_masker.py checks every one).
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

from .nl import normalize_for_matching
from .schema.catalog import quote_identifier

#: One word: letters, digits and Arabic diacritics, with hyphens and apostrophes allowed
#: between letters ("Ali-Baba" is one word, so "Ali" is not inside it).
_WORD_RE = re.compile(r"(?:[\wً-ٰٟ]|(?<=\w)[-'’](?=\w))+")

#: An English possessive on the last word of a name ("Al Harbi's"): matched without it,
#: and kept in the text after the placeholder. Found by the privacy eval, where
#: "Hessa Al Harbi's orders" went to the model unmasked.
_POSSESSIVE_RE = re.compile(r"['’]s$", re.IGNORECASE)

#: Placeholder-looking text the user typed. Issued names never reuse one of these.
_TYPED_PLACEHOLDER_RE = re.compile(r":([A-Za-z_]\w*)")

_PREFIX_RE = re.compile(r"[a-z][a-z0-9_]*")

#: Private values recognised by their shape, as (placeholder prefix, pattern). Checked
#: before names, so a name inside an email address is masked as part of the address.
_SHAPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")),
    # Pakistan, then the Gulf codes (965 Kuwait, 966 Saudi, 968 Oman, 971 UAE, 973 Bahrain,
    # 974 Qatar). International form only: a bare local number is indistinguishable from an
    # order number.
    ("phone", re.compile(r"\+(?:92|9[67]\d)(?:[ -]?\d{2,7}){2,4}(?!\d)")),
    # Emirates ID (784-YYYY-NNNNNNN-N), then Pakistani CNIC (NNNNN-NNNNNNN-N).
    ("national_id", re.compile(r"(?<!\d)(?:784-\d{4}-\d{7}-\d|\d{5}-\d{7}-\d)(?!\d)")),
)


@dataclass(frozen=True)
class Binding:
    """The real value behind one placeholder, and the column it belongs to (if any)."""

    value: str
    table: str | None = None
    column: str | None = None


@dataclass(frozen=True)
class SensitiveValue:
    """One stored private value and the placeholder prefix it is masked with."""

    value: str
    table: str
    column: str
    prefix: str


@dataclass(frozen=True)
class Masked:
    """A question with its private values replaced, and what each placeholder stands for.

    ``bindings`` maps placeholder names (without the colon) to their values, in the order
    they first appear in the text.
    """

    text: str
    bindings: dict[str, Binding] = field(default_factory=dict)


def has_typed_placeholder(text: str) -> bool:
    """Whether ``text`` contains placeholder-looking text the user typed (``:name``).

    The masker never binds those, but a shared cache keyed by masked questions must not let
    them reach another person's entry, so the cache steps aside for such a question.
    """
    return _TYPED_PLACEHOLDER_RE.search(text) is not None


def load_masker(db_path: Path) -> Masker | None:
    """The masker for ``db_path`` from ``<db>.privacy.json`` beside it, or ``None`` if absent."""
    config = db_path.with_suffix(".privacy.json")
    if not config.exists():
        return None
    return Masker.from_db(db_path, load_privacy_config(config))


def load_privacy_config(path: Path) -> dict[str, str]:
    """Read ``<db>.privacy.json``: ``{"table.column": "placeholder prefix"}``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    columns = data.get("sensitive_columns", {})
    if not isinstance(columns, dict):
        raise ValueError(f"{path}: sensitive_columns must be an object")
    for key, prefix in columns.items():
        if not isinstance(key, str) or key.count(".") != 1:
            raise ValueError(f"{path}: {key!r} is not table.column")
        if not isinstance(prefix, str) or not _PREFIX_RE.fullmatch(prefix):
            # A leading underscore is reserved for the requester's own attributes.
            raise ValueError(f"{path}: prefix {prefix!r} for {key} must match {_PREFIX_RE.pattern}")
    return dict(columns)


class Masker:
    """Replaces private values in a cleaned question with typed placeholders."""

    def __init__(self, values: Iterable[SensitiveValue]) -> None:
        # Keyed by the first word, each list longest first, so "Omar Al Thani Jr" is tried
        # before "Omar Al Thani". The first stored spelling of a key wins.
        index: dict[str, list[tuple[tuple[str, ...], SensitiveValue]]] = {}
        seen: set[tuple[str, tuple[str, ...]]] = set()
        values_seen: list[SensitiveValue] = []
        for entry in values:
            values_seen.append(entry)
            keys = tuple(_key(m.group()) for m in _WORD_RE.finditer(entry.value))
            if not keys or (entry.prefix, keys) in seen:
                continue
            seen.add((entry.prefix, keys))
            index.setdefault(keys[0], []).append((keys, entry))
        for candidates in index.values():
            candidates.sort(key=lambda item: len(item[0]), reverse=True)
        self._index = index
        self._folded = sorted({(entry.value, _key(entry.value)) for entry in values_seen})
        #: ``(table, column)`` for every column values were loaded from. Grounding never
        #: offers stored values of these columns to the model as suggestions.
        self.sensitive_columns = frozenset({(v.table, v.column) for v in values_seen})

    @classmethod
    def from_db(cls, db_path: Path, sensitive_columns: Mapping[str, str]) -> Masker:
        """Load the distinct stored values of each sensitive column, read-only."""
        values: list[SensitiveValue] = []
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            for key, prefix in sensitive_columns.items():
                table, _, column = key.partition(".")
                known = {
                    row[1] for row in conn.execute(f"PRAGMA table_info({quote_identifier(table)})")
                }
                if column not in known:
                    raise ValueError(f"privacy config names an unknown column: {key}")
                rows = conn.execute(
                    f"SELECT DISTINCT {quote_identifier(column)} FROM {quote_identifier(table)} "  # noqa: S608
                    f"WHERE {quote_identifier(column)} IS NOT NULL"
                )
                for (value,) in rows:
                    text = str(value).strip()
                    if text:
                        values.append(SensitiveValue(text, table, column, prefix))
        return cls(values)

    def mask(self, text: str, known: Mapping[str, Binding] | None = None) -> Masked:
        """Replace every private value in ``text`` (``clean_for_model`` output).

        ``known`` holds placeholders issued earlier in the same conversation: a value seen
        before keeps its name, and a new one is numbered after them, so a follow-up's SQL
        can use both. ``bindings`` holds only the placeholders that occur in ``text``.
        """
        hits: list[tuple[int, int, str, Binding]] = []
        for prefix, pattern in _SHAPES:
            for m in pattern.finditer(text):
                if not _overlaps(hits, m.start(), m.end()):
                    hits.append((m.start(), m.end(), prefix, Binding(m.group())))

        words = [m for m in _WORD_RE.finditer(text) if not _overlaps(hits, m.start(), m.end())]
        i = 0
        while i < len(words):
            matched = self._match_at(text, words, i)
            if matched is None:
                i += 1
                continue
            length, entry, end = matched
            start = words[i].start()
            hits.append((start, end, entry.prefix, Binding(entry.value, entry.table, entry.column)))
            i += length

        if not hits:
            return Masked(text)
        return _replace(text, sorted(hits, key=lambda hit: hit[0]), known or {})

    def find_values(self, text: str) -> list[str]:
        """Stored private values that occur in ``text``, in any spelling (the leak check).

        Compared after ``normalize_for_matching`` as well as verbatim, so a value sent with
        another case, without the madda or with Urdu letters still counts. This is how the
        eval counts what reached the model (``private_values_sent``).
        """
        folded = _key(text)
        return [value for value, key in self._folded if value in text or key in folded]

    def _match_at(
        self, text: str, words: list[re.Match[str]], i: int
    ) -> tuple[int, SensitiveValue, int] | None:
        """``(words matched, value, end offset)`` of the longest value starting at word ``i``.

        The end offset stops before a possessive on the last word, so "Al Harbi's" becomes
        ":customer_name_1's".
        """
        first = words[i].group()
        candidates = [*self._index.get(_key(first), ())]
        if (stem := _POSSESSIVE_RE.sub("", first)) != first:
            candidates += self._index.get(_key(stem), ())
        for keys, entry in sorted(candidates, key=lambda item: len(item[0]), reverse=True):
            n = len(keys)
            span = words[i : i + n]
            if len(span) < n:
                continue
            *body, last = span
            if any(_key(w.group()) != k for w, k in zip(body, keys, strict=False)):
                continue
            word = last.group()
            if _key(word) == keys[-1]:
                end = last.end()
            elif _key(stripped := _POSSESSIVE_RE.sub("", word)) == keys[-1]:
                end = last.start() + len(stripped)
            else:
                continue
            # The words of a name are separated by spaces only: "Tariq, Al Thani" is a list.
            if all(text[a.end() : b.start()].isspace() for a, b in pairwise(span)):
                return n, entry, end
        return None


def _key(word: str) -> str:
    return normalize_for_matching(word)


def _overlaps(hits: list[tuple[int, int, str, Binding]], start: int, end: int) -> bool:
    return any(start < h_end and h_start < end for h_start, h_end, _, _ in hits)


def _replace(
    text: str, hits: list[tuple[int, int, str, Binding]], known: Mapping[str, Binding]
) -> Masked:
    """Rebuild the text with numbered placeholders, one name per distinct value."""
    taken = {m.group(1) for m in _TYPED_PLACEHOLDER_RE.finditer(text)} | set(known)
    counters: dict[str, int] = {}
    names = {(name.rpartition("_")[0], binding): name for name, binding in known.items()}
    bindings: dict[str, Binding] = {}
    parts: list[str] = []
    cursor = 0
    for start, end, prefix, binding in hits:
        name = names.get((prefix, binding))
        if name is not None:
            bindings.setdefault(name, binding)
        else:
            n = counters.get(prefix, 0)
            while True:
                n += 1
                name = f"{prefix}_{n}"
                if name not in taken:
                    break
            counters[prefix] = n
            names[(prefix, binding)] = name
            bindings[name] = binding
        parts += [text[cursor:start], f":{name}"]
        cursor = end
    parts.append(text[cursor:])
    return Masked("".join(parts), bindings)
