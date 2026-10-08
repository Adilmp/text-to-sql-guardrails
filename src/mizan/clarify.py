"""Clarifying questions: ask back when a business term has more than one meaning.

"How many orders did each warehouse fulfil?" counts every order to one person and only the
delivered ones to another; the eval showed the model reading the Urdu verb (پورے کیے) as
"delivered" while the English question read as "all". Guessing either way is wrong half the
time, so the server asks: "Which orders should count?", with the options in the question's
language. The choice is added to the question as a short English hint for the model
("… (count all orders, whatever their status)") and the question is answered as usual.

Why a curated list and not the model
------------------------------------
Asking the model to decide when to ask would change every prompt, and a 7b model asked to be
careful tends to ask about everything. A short list of terms known to be ambiguous in this
data, with the words that already settle them ("units" settles "best-selling"), is
predictable, costs no generation, and is checked against every eval question in the tests.
It lives in ``<db>.clarifications.json`` next to the glossary: what is ambiguous is a fact
about this business's vocabulary, not about the code.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .nl import arabic_ratio, detect_script, normalize_for_matching

_NON_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)


@dataclass(frozen=True)
class Option:
    #: What the user sees, per language.
    label: dict[str, str]
    #: Added to the question for the model, in English.
    hint: str


@dataclass(frozen=True)
class Ambiguity:
    id: str
    terms: tuple[str, ...]
    unless: tuple[str, ...]
    ask: dict[str, str]
    options: tuple[Option, ...]

    def to_dict(self, language: str) -> dict[str, object]:
        """What the page shows, in the question's language (English when there is none)."""
        return {
            "id": self.id,
            "ask": self.ask.get(language) or self.ask["en"],
            "options": [o.label.get(language) or o.label["en"] for o in self.options],
        }


def _prepare(term: str) -> tuple[str, bool]:
    """The folded term, and whether it may also start a longer word.

    Arabic attaches pronouns and suffixes (نفذ → نفذها), and English inflects long verbs
    (fulfil → fulfilled), so those match as a prefix. A short Latin term ("sent") must be a
    whole word, or it would match "sentence".
    """
    folded = _NON_WORD_RE.sub(" ", normalize_for_matching(term)).strip()
    return folded, arabic_ratio(term) > 0.5 or len(folded) >= 5


def _contains(text: str, term: str, prefix: bool) -> bool:
    tail = "" if prefix else r"(?!\w)"
    return re.search(rf"(?<!\w){re.escape(term)}{tail}", text) is not None


class Clarifier:
    """Finds the first ambiguous term in a question that nothing else in it settles."""

    def __init__(self, ambiguities: tuple[Ambiguity, ...] = ()) -> None:
        self.ambiguities = ambiguities
        # Words that settle the meaning match generously, as the start of any word ("unit"
        # settles "units"): a needless question costs more than a missed one.
        self._prepared = [
            (a, [_prepare(t) for t in a.terms], [(_prepare(t)[0], True) for t in a.unless])
            for a in ambiguities
        ]

    @classmethod
    def from_file(cls, path: Path) -> Clarifier:
        """Load ``<db>.clarifications.json``; a missing file means nothing is ambiguous."""
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        found = []
        for raw in data.get("ambiguities", []):
            terms = tuple(t for words in raw["terms"].values() for t in words)
            unless = tuple(t for words in raw.get("unless", {}).values() for t in words)
            options = tuple(Option(dict(o["label"]), str(o["hint"])) for o in raw["options"])
            if len(options) < 2:
                raise ValueError(f"ambiguity {raw['id']!r} needs at least two options")
            found.append(Ambiguity(raw["id"], terms, unless, dict(raw["ask"]), options))
        return cls(tuple(found))

    def check(self, question: str) -> Ambiguity | None:
        text = _NON_WORD_RE.sub(" ", normalize_for_matching(question)).strip()
        for ambiguity, terms, unless in self._prepared:
            if any(_contains(text, t, p) for t, p in terms if t) and not any(
                _contains(text, u, p) for u, p in unless if u
            ):
                return ambiguity
        return None

    def get(self, ambiguity_id: str) -> Ambiguity | None:
        return next((a for a in self.ambiguities if a.id == ambiguity_id), None)

    def apply(self, question: str, ambiguity_id: str, option: int) -> tuple[str, str]:
        """The question with the chosen meaning added, and the option's label to show.

        Raises ``ValueError`` for an unknown ambiguity or option: the choice comes from the
        browser, so it is checked, and only the server's own hint text reaches the model.
        """
        ambiguity = self.get(ambiguity_id)
        if ambiguity is None or not 0 <= option < len(ambiguity.options):
            raise ValueError("unknown clarification")
        chosen = ambiguity.options[option]
        language = detect_script(question).prompt_language
        label = chosen.label.get(language) or chosen.label["en"]
        return f"{question.rstrip()} ({chosen.hint})", label
