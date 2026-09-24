"""Catalog introspection, glossary merging and prompt rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mizan.db.build_synthetic import build
from mizan.errors import SchemaError
from mizan.nl import Script
from mizan.schema import Catalog


class TestIntrospection:
    def test_all_tables_found(self, catalog: Catalog) -> None:
        assert catalog.all_table_names == {
            "customers",
            "orders",
            "order_items",
            "products",
            "warehouses",
            "couriers",
        }

    def test_foreign_keys(self, catalog: Catalog) -> None:
        targets = {fk.references_table for fk in catalog.table("orders").foreign_keys}
        assert targets == {"customers", "warehouses", "couriers"}

    def test_has_column_is_case_insensitive(self, catalog: Catalog) -> None:
        assert catalog.has_column("orders", "STATUS")
        assert not catalog.has_column("orders", "not_a_column")

    def test_unknown_table_raises(self, catalog: Catalog) -> None:
        with pytest.raises(SchemaError):
            catalog.table("nope")


class TestSampleValues:
    def test_low_cardinality_column_gets_samples(self, catalog: Catalog) -> None:
        """This is what stops the model inventing status='shipped'."""
        status = catalog.table("orders").column("status")
        assert status is not None
        assert set(status.sample_values) == {
            "pending",
            "in_transit",
            "delivered",
            "returned",
            "cancelled",
        }

    def test_high_cardinality_column_gets_no_samples(self, catalog: Catalog) -> None:
        """Customer names would leak data into the prompt and help nothing."""
        name = catalog.table("customers").column("name_en")
        assert name is not None
        assert name.sample_values == ()

    def test_numeric_column_gets_no_samples(self, catalog: Catalog) -> None:
        total = catalog.table("orders").column("total_aed")
        assert total is not None
        assert total.sample_values == ()


class TestGlossary:
    def test_arabic_aliases_applied(self, catalog: Catalog) -> None:
        assert "الطلبات" in catalog.table("orders").aliases_ar  # "the orders"

    def test_arabic_index_is_normalized(self, catalog: Catalog) -> None:
        """Lookup works regardless of how the alias was spelled."""
        from mizan.nl import normalize_for_matching

        index = catalog.arabic_index()
        assert index[normalize_for_matching("الطلبات")] == ("orders", None)
        assert index[normalize_for_matching("الكمية")] == ("order_items", "quantity")

    def test_glossary_with_unknown_table_fails_loudly(
        self, catalog: Catalog, tmp_path: Path
    ) -> None:
        """Silent drift would degrade Arabic matching invisibly."""
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"tables": {"ghost_table": {"ar": ["x"]}}}), encoding="utf-8")
        with pytest.raises(SchemaError, match="unknown table"):
            catalog.apply_glossary(bad)

    def test_glossary_with_unknown_column_fails_loudly(
        self, catalog: Catalog, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text(
            json.dumps({"tables": {"orders": {"columns": {"ghost_col": {"ar": ["x"]}}}}}),
            encoding="utf-8",
        )
        with pytest.raises(SchemaError, match="unknown columns"):
            catalog.apply_glossary(bad)


class TestPromptRendering:
    def test_renders_ddl(self, catalog: Catalog) -> None:
        prompt = catalog.to_prompt(include_arabic=False)
        assert "CREATE TABLE orders" in prompt
        assert "FOREIGN KEY" in prompt

    def test_arabic_aliases_only_when_requested(self, catalog: Catalog) -> None:
        assert "الطلبات" in catalog.to_prompt(include_arabic=True)
        assert "الطلبات" not in catalog.to_prompt(include_arabic=False)

    def test_sample_values_rendered(self, catalog: Catalog) -> None:
        assert "'in_transit'" in catalog.to_prompt(include_arabic=False)

    def test_system_prompt_language_switches(self, catalog: Catalog) -> None:
        from mizan.generate import build_system_prompt

        assert "القواعد" in build_system_prompt(catalog, Script.ARABIC)
        assert "Rules:" in build_system_prompt(catalog, Script.ENGLISH)


class TestDeterminism:
    def test_same_seed_produces_identical_database(self, tmp_path: Path) -> None:
        """Eval numbers are only meaningful if the data is reproducible."""
        a = build(tmp_path / "a.sqlite", n_customers=20, n_orders=50)
        b = build(tmp_path / "b.sqlite", n_customers=20, n_orders=50)
        assert a.read_bytes() == b.read_bytes()
