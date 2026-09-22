"""Schema introspection and the bilingual glossary."""

from __future__ import annotations

from pathlib import Path

from .catalog import Catalog, Column, ForeignKey, Table

__all__ = ["Catalog", "Column", "ForeignKey", "Table", "load_catalog"]


def load_catalog(db_path: Path, glossary_path: Path | None = None) -> Catalog:
    """Introspect ``db_path`` and layer the bilingual glossary on top.

    The glossary defaults to ``<db stem>.glossary.json`` beside the database, so a rebuilt
    database automatically picks its curated Arabic metadata back up.
    """
    catalog = Catalog.from_sqlite(db_path)
    path = glossary_path or db_path.with_suffix(".glossary.json")
    catalog.apply_glossary(path)
    return catalog
