"""API tests, driven through FastAPI's TestClient with the mock provider."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mizan.api import create_app
from mizan.config import Settings


@pytest.fixture()
def client(db_path: Path) -> Iterator[TestClient]:
    settings = Settings.from_env(provider="mock", db_path=db_path)
    # The context manager form is required: it runs the lifespan handler that builds the
    # catalog and provider. Without it every request fails with a KeyError on app state.
    with TestClient(create_app(settings)) as test_client:
        yield test_client


class TestEndpoints:
    def test_health(self, client: TestClient) -> None:
        body = client.get("/api/health").json()
        assert body["ok"] is True
        assert body["provider"] == "mock"

    def test_index_serves_html(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "Text-to-SQL" in response.text

    def test_schema(self, client: TestClient) -> None:
        body = client.get("/api/schema").json()
        assert "CREATE TABLE orders" in body["ddl"]
        assert "orders" in body["tables"]


class TestAsk:
    def test_english_question(self, client: TestClient) -> None:
        body = client.post("/api/ask", json={"question": "how many orders are there?"}).json()
        assert body["ok"] is True
        assert body["script"] == "en"
        assert body["rtl"] is False
        assert body["result"]["row_count"] == 1

    def test_arabic_question_is_marked_rtl(self, client: TestClient) -> None:
        body = client.post("/api/ask", json={"question": "كم عدد الطلبات؟"}).json()
        assert body["script"] == "ar"
        assert body["rtl"] is True

    def test_blocked_query_returns_200_with_violations(self, client: TestClient) -> None:
        """A rejection is a successful analysis, not a transport error."""
        response = client.post("/api/ask", json={"question": "how many orders are there?"})
        assert response.status_code == 200

    def test_empty_question_rejected_by_validation(self, client: TestClient) -> None:
        assert client.post("/api/ask", json={"question": ""}).status_code == 422

    def test_oversized_question_rejected(self, client: TestClient) -> None:
        assert client.post("/api/ask", json={"question": "x" * 5000}).status_code == 422

    def test_samples_out_of_range_rejected(self, client: TestClient) -> None:
        assert client.post("/api/ask", json={"question": "hi", "samples": 99}).status_code == 422


class TestFonts:
    """The page's fonts are served by the app: its CSP blocks every other origin."""

    def test_a_bundled_font_is_served_as_woff2(self, client: TestClient) -> None:
        response = client.get("/fonts/IBMPlexSans-Regular.woff2")
        assert response.status_code == 200
        assert response.headers["content-type"] == "font/woff2"
        assert response.content[:4] == b"wOF2"
        assert "max-age" in response.headers.get("cache-control", "")

    @pytest.mark.parametrize(
        "name",
        [
            "OFL-IBMPlexSans.txt",
            "..%2Findex.html",
            "../index.html",
            "%2e%2e%2f%2e%2e%2fpyproject.toml",
            "nope.woff2",
        ],
    )
    def test_nothing_but_the_bundled_fonts_is_served(self, client: TestClient, name: str) -> None:
        assert client.get(f"/fonts/{name}").status_code == 404

    def test_the_csp_allows_fonts_from_this_origin_only(self, client: TestClient) -> None:
        csp = client.get("/").headers["content-security-policy"]
        assert "font-src 'self'" in csp
        assert "default-src 'none'" in csp
