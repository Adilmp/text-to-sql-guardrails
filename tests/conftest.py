"""Shared fixtures.

The database is built once per test session into a temporary directory. It is deterministic
(seeded), so tests can assert on exact row counts without being brittle.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mizan.config import Settings
from mizan.db.build_synthetic import build
from mizan.schema import Catalog, load_catalog

REPO_ROOT = Path(__file__).resolve().parent.parent
GLOSSARY = REPO_ROOT / "data" / "gulf_logistics.glossary.json"


@pytest.fixture(scope="session")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A freshly built demo database, shared across the session."""
    path = tmp_path_factory.mktemp("mizan-db") / "test.sqlite"
    build(path, n_customers=40, n_orders=120)
    return path


@pytest.fixture(scope="session")
def catalog(db_path: Path) -> Catalog:
    return load_catalog(db_path, GLOSSARY)


@pytest.fixture()
def settings(db_path: Path) -> Settings:
    return Settings.from_env(provider="mock", db_path=db_path, max_rows=50)
