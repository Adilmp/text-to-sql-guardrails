"""The answer cache: what counts as the same question, what gets stored, and what never does.

Offline: the model is a scripted ``MockProvider``, so a test can count exactly how many
questions reached it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mizan.api import create_app
from mizan.cache import (
    AnswerCache,
    CachedTextToSQL,
    CacheSettings,
    cache_context,
    canonical_question,
)
from mizan.config import Settings
from mizan.generate import TextToSQL
from mizan.providers import MockProvider
from mizan.schema import Catalog

LATE = "SELECT COUNT(*) FROM orders WHERE delivered_at IS NOT NULL AND delivered_at > promised_at"
EARLY = "SELECT COUNT(*) FROM orders WHERE delivered_at IS NOT NULL AND delivered_at <= promised_at"


class TestCanonicalQuestion:
    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("How many orders were delivered late?", "how many orders were delivered late"),
            (
                "how many orders were delivered late?",
                "Please, how many orders were delivered LATE ?",
            ),
            ("how many customers are in Dubai?", "Can you tell me how many customers are in Dubai"),
            ("كم عدد الطلبات؟", "كَمْ عَدَدُ الطّلبات ؟"),  # diacritics
            ("كم عدد الطلبات؟", "من فضلك كم عدد الطلبات؟"),  # "please"
            ("کتنے آرڈر ہیں؟", "كتني آرڈر هيں"),  # Urdu typed on an Arabic keyboard
            ("کتنے آرڈر ہیں؟", "براہ کرم کتنے آرڈر ہیں؟"),  # Urdu "please"
            ("how many orders have a total above ٥٠٠?", "how many orders have a total above 500"),
        ],
    )
    def test_meaning_free_variation_shares_a_key(self, a: str, b: str) -> None:
        assert canonical_question(a) == canonical_question(b)

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            # Each pair scored 0.95-1.00 with the local embedding model (D40): exactly the
            # differences a similarity cache would have collapsed.
            ("how many orders were delivered late?", "how many orders were delivered early?"),
            ("how many customers are in Dubai?", "how many customers are in Riyadh?"),
            ("which courier was late most often?", "which courier was late least often?"),
            ("how many orders are above 500?", "how many orders are above 600?"),
            ("which couriers are not active?", "which couriers are active?"),
            ("how many orders were placed in 2025?", "how many orders were delivered in 2025?"),
        ],
    )
    def test_any_meaningful_difference_keeps_keys_apart(self, a: str, b: str) -> None:
        assert canonical_question(a) != canonical_question(b)


def served(
    catalog: Catalog, settings: Settings, tmp_path: Path, *script: str
) -> tuple[CachedTextToSQL, MockProvider]:
    provider = MockProvider(script=list(script))
    engine = TextToSQL(catalog, provider, settings)
    front = CachedTextToSQL(engine, AnswerCache(tmp_path / "c.sqlite"), CacheSettings())
    return front, provider


class TestCachedAnswers:
    def test_a_repeat_is_served_without_the_model(
        self, catalog: Catalog, settings: Settings, tmp_path: Path
    ) -> None:
        front, provider = served(catalog, settings, tmp_path, LATE)
        first, hit = front.ask("How many orders were delivered late?")
        assert first.ok and hit is None
        again, hit = front.ask("how many orders were delivered late")
        assert again.ok and hit is not None and hit.hits == 1
        assert hit.question == "How many orders were delivered late?"
        assert again.result is not None and first.result is not None
        assert again.result.rows == first.result.rows
        assert len(provider.calls) == 1  # the model was asked once

    def test_a_different_question_goes_to_the_model(
        self, catalog: Catalog, settings: Settings, tmp_path: Path
    ) -> None:
        front, provider = served(catalog, settings, tmp_path, LATE, EARLY)
        front.ask("how many orders were delivered late?")
        _, hit = front.ask("how many orders were delivered early?")
        assert hit is None and len(provider.calls) == 2

    def test_a_repaired_answer_is_not_cached(
        self, catalog: Catalog, settings: Settings, tmp_path: Path
    ) -> None:
        bad = "SELECT city, AVG(total_aed) FROM orders GROUP BY city"
        good = (
            "SELECT c.city, AVG(o.total_aed) FROM orders o "
            "JOIN customers c ON c.customer_id = o.customer_id GROUP BY c.city"
        )
        front, provider = served(catalog, settings, tmp_path, bad, good, good)
        first, _ = front.ask("average order total per customer city?")
        assert first.ok and first.repairs == 1
        _, hit = front.ask("average order total per customer city?")
        assert hit is None and len(provider.calls) == 3

    def test_a_blocked_answer_is_not_cached(
        self, catalog: Catalog, settings: Settings, tmp_path: Path
    ) -> None:
        front, provider = served(catalog, settings, tmp_path, "DROP TABLE orders", LATE)
        assert not front.ask("drop the orders table")[0].ok
        front.ask("drop the orders table")
        assert len(provider.calls) == 2

    def test_a_tampered_entry_is_validated_and_evicted(
        self, catalog: Catalog, settings: Settings, tmp_path: Path
    ) -> None:
        """The cache file is untrusted: planted SQL must never run."""
        front, provider = served(catalog, settings, tmp_path, LATE)
        assert front.cache is not None
        key = canonical_question("how many orders were delivered late?")
        front.cache.put(front.context, key, "planted", "DROP TABLE orders", 1.0)
        answer, hit = front.ask("how many orders were delivered late?")
        assert hit is None and answer.ok  # answered by the model instead
        assert len(provider.calls) == 1
        assert front.cache.get(front.context, key) is not None  # replaced by the real answer

    def test_entries_do_not_survive_a_prompt_or_model_change(
        self, catalog: Catalog, settings: Settings, tmp_path: Path
    ) -> None:
        engine = TextToSQL(catalog, MockProvider(), settings)
        other_model = TextToSQL(catalog, MockProvider(model="mock-2"), settings)
        assert cache_context(engine) != cache_context(other_model)

    def test_disabled_cache_always_asks(
        self, catalog: Catalog, settings: Settings, tmp_path: Path
    ) -> None:
        provider = MockProvider(script=[LATE, LATE])
        front = CachedTextToSQL(
            TextToSQL(catalog, provider, settings),
            AnswerCache(tmp_path / "c.sqlite"),
            CacheSettings(enabled=False),
        )
        front.ask("how many orders were delivered late?")
        _, hit = front.ask("how many orders were delivered late?")
        assert hit is None and len(provider.calls) == 2

    def test_settings_from_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("MIZAN_CACHE", "off")
        monkeypatch.setenv("MIZAN_CACHE_PATH", str(tmp_path / "x.sqlite"))
        cfg = CacheSettings.from_env()
        assert cfg.enabled is False and cfg.path == tmp_path / "x.sqlite"


@pytest.fixture()
def cached_client(db_path: Path, tmp_path: Path) -> Iterator[TestClient]:
    settings = Settings.from_env(provider="mock", db_path=db_path)
    app = create_app(settings, CacheSettings(path=tmp_path / "api.sqlite"))
    with TestClient(app) as test_client:
        yield test_client


class TestApi:
    def test_second_ask_reports_the_cache_hit(self, cached_client: TestClient) -> None:
        question = {"question": "how many orders were delivered late?"}
        first = cached_client.post("/api/ask", json=question).json()
        assert first["ok"] and first["cache"] is None
        second = cached_client.post(
            "/api/ask", json={"question": "How many orders were delivered late"}
        ).json()
        assert second["cache"]["hit"] is True
        assert second["cache"]["matched"] == "how many orders were delivered late?"
        assert second["result"]["rows"] == first["result"]["rows"]
        assert second["result"]["columns"] == first["result"]["columns"]
        health = cached_client.get("/api/health").json()
        assert health["cached_answers"] == 1 and health["warming_up"] is False

    def test_samples_bypass_the_cache(self, cached_client: TestClient) -> None:
        question = {"question": "how many orders were delivered late?"}
        cached_client.post("/api/ask", json=question)
        sampled = cached_client.post("/api/ask", json={**question, "samples": 2}).json()
        assert sampled["cache"] is None


class TestKeepWarm:
    def test_the_keep_warm_interval_is_inside_ollama_s_keep_alive(self) -> None:
        """A ping every N seconds only keeps the model loaded if N < the keep-alive."""
        from mizan.api import KEEP_WARM_EVERY_S

        keep_alive = Settings.from_env().ollama_keep_alive
        assert keep_alive.endswith("m")
        assert int(keep_alive[:-1]) * 60 > KEEP_WARM_EVERY_S

    def test_no_bare_keep_alive_requests(self) -> None:
        """Regression: a keep-alive request without the loaded context size made Ollama
        reload the model and drop the processed prompt. The server must only talk to the
        model through the provider."""
        source = (Path(__file__).resolve().parent.parent / "src/mizan/api.py").read_text(
            encoding="utf-8"
        )
        assert "/api/generate" not in source and "httpx" not in source


class TestPrefill:
    """The curated question list behind pre-filling and suggestions (mizan/questions.py)."""

    def test_every_language_every_value_and_no_repeats(self) -> None:
        from mizan.questions import EXAMPLES, curated_questions

        names = {
            "couriers": [("Gulf Express", "الخليج السريع")],
            "warehouses": [("Dubai Hub", "مستودع دبي")],
            "products": [("Prayer Rug", "سجادة صلاة")],
        }
        texts = [q for _, q in curated_questions(names)]
        keys = [canonical_question(q) for q in texts]
        assert len(keys) == len(set(keys))  # no question is asked twice
        assert texts[: len(EXAMPLES)] == list(EXAMPLES)  # most likely first
        assert "how many customers are in Riyadh?" in texts
        assert "كم عدد العملاء في الرياض؟" in texts
        assert "ریاض میں کتنے کسٹمرز ہیں؟" in texts
        assert "how many orders did Gulf Express deliver late?" in texts

    def test_gold_answers_are_never_planted(self) -> None:
        """Pre-filling asks the model; nothing on that path touches the eval's gold SQL."""
        root = Path(__file__).resolve().parent.parent
        for path in (root / "scripts" / "prefill_cache.py", root / "src/mizan/questions.py"):
            source = path.read_text(encoding="utf-8")
            assert "gold_sql" not in source and "gold_queries" not in source

    def test_has_does_not_count_a_hit(self, tmp_path: Path) -> None:
        cache = AnswerCache(tmp_path / "c.sqlite")
        assert not cache.has("ctx", "k")
        cache.put("ctx", "k", "q", "SELECT 1", 1.0)
        assert cache.has("ctx", "k") and cache.has("ctx", "k")
        entry = cache.get("ctx", "k")
        assert entry is not None and entry[2] == 1  # first counted hit
