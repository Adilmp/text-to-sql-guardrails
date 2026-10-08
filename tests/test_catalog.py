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

    def test_urdu_aliases_applied(self, catalog: Catalog) -> None:
        assert "آرڈرز" in catalog.table("orders").aliases_ur  # "orders"

    def test_alias_index_matches_urdu_typed_on_either_keyboard(self, catalog: Catalog) -> None:
        """کسٹمرز typed with Arabic kaf (ك) and yeh (ي) still resolves."""
        from mizan.nl import normalize_for_matching

        index = catalog.alias_index()
        assert index[normalize_for_matching("کسٹمرز")] == ("customers", None)
        assert index[normalize_for_matching("كسٹمرز")] == ("customers", None)

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
        prompt = catalog.to_prompt()
        assert "CREATE TABLE orders" in prompt
        assert "FOREIGN KEY" in prompt

    def test_schema_card_is_valid_ddl(self, catalog: Catalog) -> None:
        """The card must be real DDL, comments and all: SQLite has to accept it.

        Regression: the comma separating column definitions used to be appended after
        each column's ``--`` comment, so it ended up inside the comment.
        """
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.executescript(catalog.to_prompt(languages=("ar", "ur")))
        created = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
        assert created == set(catalog.all_table_names)

    def test_aliases_only_for_requested_languages(self, catalog: Catalog) -> None:
        assert "الطلبات" in catalog.to_prompt(languages=("ar",))
        assert "الطلبات" not in catalog.to_prompt()
        assert "آرڈرز" in catalog.to_prompt(languages=("ur",))
        assert "آرڈرز" not in catalog.to_prompt(languages=("ar",))

    def test_sample_values_rendered(self, catalog: Catalog) -> None:
        assert "'in_transit'" in catalog.to_prompt()

    def test_repeated_names_are_marked_not_unique(self, catalog: Catalog) -> None:
        """Profiled from the data: grouping by a name that repeats merges different people."""
        name = catalog.table("customers").column("name_en")
        assert name is not None and name.unique is False
        assert "NOT unique" in catalog.to_prompt()
        # Dates and keys are not profiled: "are these timestamps distinct?" changes nothing.
        placed = catalog.table("orders").column("placed_at")
        assert placed is not None and placed.unique is None

    def test_join_conditions_come_from_foreign_keys(self, catalog: Catalog) -> None:
        joins = catalog.join_conditions()
        assert "orders.customer_id = customers.customer_id" in joins
        assert "order_items.product_id = products.product_id" in joins

    def test_system_prompt_is_identical_for_every_language(self, catalog: Catalog) -> None:
        """One shared prefix is what lets the model server reuse the processed prompt
        whatever language the previous question was in (D33)."""
        from mizan.generate import build_system_prompt

        prompts = {build_system_prompt(catalog, script) for script in Script}
        assert len(prompts) == 1
        prompt = prompts.pop()
        assert "Join conditions:" in prompt and "Definitions:" in prompt
        assert "الطلبات" in prompt and "آرڈرز" in prompt


class TestDefinitions:
    def test_definitions_loaded_from_glossary(self, catalog: Catalog) -> None:
        terms = {d.term for d in catalog.definitions}
        assert "late order" in terms
        assert any("order_items.unit_price_aed" in d.sql for d in catalog.definitions)

    def test_definition_with_unknown_column_fails_loudly(
        self, catalog: Catalog, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text(
            json.dumps({"definitions": [{"term": "x", "sql": "SUM(orders.ghost)"}]}),
            encoding="utf-8",
        )
        before = (dict(catalog.tables), catalog.definitions)
        with pytest.raises(SchemaError, match="unknown column"):
            catalog.apply_glossary(bad)
        # A failed glossary leaves the catalog untouched.
        assert (catalog.tables, catalog.definitions) == before


class TestDeterminism:
    def test_same_seed_produces_identical_database(self, tmp_path: Path) -> None:
        """Eval numbers are only meaningful if the data is reproducible."""
        a = build(tmp_path / "a.sqlite", n_customers=20, n_orders=50)
        b = build(tmp_path / "b.sqlite", n_customers=20, n_orders=50)
        assert a.read_bytes() == b.read_bytes()
