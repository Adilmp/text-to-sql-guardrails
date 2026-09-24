"""Deterministic in-process provider.

This is the reason the test suite runs offline in under a second, and it is doing more work
than "return a fixed string":

* **Fixture matching is Arabic-aware.** Lookup keys go through
  :func:`normalize_for_matching`, so a test can ask the same question with or without
  diacritics and get the same fixture. That means the mock exercises the normalization
  pipeline rather than bypassing it.
* **It can be scripted to misbehave.** ``MockProvider(script=[...])`` returns a fixed
  sequence of responses, which is how the guardrail and confidence tests provoke specific
  failures — a hallucinated column, a stacked query, a Markdown-fenced answer — without
  needing a model that happens to make that mistake today.
* **It records every call.** ``calls`` lets tests assert on what the prompt actually
  contained, catching regressions where a schema card silently stops being included.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..errors import ProviderResponseError
from ..nl.normalize import normalize_for_matching
from .base import Completion, Provider


@dataclass
class RecordedCall:
    system: str
    user: str
    temperature: float
    max_tokens: int


#: Question -> SQL. Keys are stored normalized; see ``_key``.
DEFAULT_FIXTURES: dict[str, str] = {
    "how many orders are there?": "SELECT COUNT(*) FROM orders",
    "كم عدد الطلبات؟": "SELECT COUNT(*) FROM orders",
    "how many orders per status?": (
        "SELECT status, COUNT(*) AS n FROM orders GROUP BY status ORDER BY n DESC"
    ),
    "how many orders were delivered late?": (
        "SELECT COUNT(*) FROM orders WHERE delivered_at IS NOT NULL AND delivered_at > promised_at"
    ),
    # Keyed on "courier delivered late" so both the singular and plural phrasings of the
    # demo question match. Fixture keys that are narrower than the questions they are meant
    # to answer fall through to `default`, which produces a confident *wrong* answer — the
    # worst possible demo failure, because nothing on screen indicates anything went wrong.
    "courier delivered late": (
        "SELECT c.name_en, COUNT(*) AS late_count "
        "FROM orders o JOIN couriers c ON c.courier_id = o.courier_id "
        "WHERE o.delivered_at IS NOT NULL AND o.delivered_at > o.promised_at "
        "GROUP BY c.name_en ORDER BY late_count DESC"
    ),
    "كم عدد الطلبات المتأخرة في دبي؟": (
        "SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
        "WHERE c.city = 'Dubai' AND o.delivered_at > o.promised_at"
    ),
    "ما متوسط قيمة الطلب لكل مدينة؟": (
        "SELECT c.city, AVG(o.total_aed) AS avg_total FROM orders o "
        "JOIN customers c ON c.customer_id = o.customer_id GROUP BY c.city"
    ),
    # Deliberately unsafe responses, so the offline demo can exercise the rejection path.
    # A real model has to be talked into these; the mock just hands them over, which is
    # the point — what is being demonstrated is the guardrail, not the model's virtue.
    "drop the orders table": "DROP TABLE orders",
    "delete every customer": "DELETE FROM customers",
    "attach the database at /tmp/evil.db": "ATTACH DATABASE '/tmp/evil.db' AS evil",
    "read the file /etc/passwd": "SELECT readfile('/etc/passwd')",
}


def _key(text: str) -> str:
    return normalize_for_matching(text)


class MockProvider(Provider):
    name = "mock"

    def __init__(
        self,
        model: str = "mock-1",
        *,
        fixtures: dict[str, str] | None = None,
        script: list[str] | None = None,
        # Deliberately self-describing. The previous default was `SELECT COUNT(*) FROM
        # orders`, which returns a plausible number for *any* unmatched question and is
        # therefore indistinguishable from a correct answer. A default that announces
        # itself turns a silent wrong answer into an obvious one.
        default: str = "SELECT 'no mock fixture for this question' AS mock_fallback",
        fail_with: Exception | None = None,
    ) -> None:
        super().__init__(model, timeout_s=1.0, max_retries=0)
        source = DEFAULT_FIXTURES if fixtures is None else fixtures
        self.fixtures = {_key(k): v for k, v in source.items()}
        self.script = list(script) if script else None
        self.default = default
        self.fail_with = fail_with
        self.calls: list[RecordedCall] = []

    def _generate_once(
        self, system: str, user: str, *, temperature: float, max_tokens: int
    ) -> Completion:
        self.calls.append(RecordedCall(system, user, temperature, max_tokens))

        if self.fail_with is not None:
            raise self.fail_with

        if self.script is not None:
            if not self.script:
                raise ProviderResponseError("mock script exhausted", calls=len(self.calls))
            text = self.script.pop(0)
        else:
            text = self._match_fixture(user)

        # A tiny sleep keeps latency numbers non-zero so that code formatting or dividing
        # by latency does not hit a surprising zero in tests.
        time.sleep(0.001)
        return Completion(
            text=text,
            model=self.model,
            latency_ms=1.0,
            prompt_tokens=len(system.split()) + len(user.split()),
            completion_tokens=len(text.split()),
            meta={"mock": True},
        )

    def _match_fixture(self, user_prompt: str) -> str:
        """Find the fixture whose question appears in the prompt.

        The prompt the provider receives is wrapped (``Question: ...\\nSQL:``), so an exact
        dictionary lookup on the whole string never matches. Containment on the normalized
        form is used instead, and the longest matching key wins so that a specific fixture
        is not shadowed by a shorter one that happens to be a prefix of it.
        """
        haystack = _key(user_prompt)
        matches = [key for key in self.fixtures if key and key in haystack]
        if not matches:
            return self.default
        return self.fixtures[max(matches, key=len)]

    def health(self) -> bool:
        return self.fail_with is None
