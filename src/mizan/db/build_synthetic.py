"""Deterministic generator for the demo database.

Why synthetic rather than a public dataset
-------------------------------------------
The demo database has to satisfy three requirements at once that no off-the-shelf dataset
meets: it must contain *Arabic text in the data itself* (not just Arabic questions about
English data), it must exercise the query shapes that make Text-to-SQL interesting
(multi-table joins, date arithmetic, late-delivery logic, aggregation with HAVING), and it
must be small enough to commit to a repository.

Why seeded and fully deterministic
-----------------------------------
Every eval number in the README is meaningless if the database differs between runs. The
generator is seeded once, uses no wall-clock time, and derives all dates from a fixed
epoch, so ``build_synthetic.py`` produces a byte-identical database on any machine. This is
the difference between "82% execution accuracy" being a measurement and being an anecdote.
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from ..logging import get_logger

logger = get_logger("db.synthetic")

#: Fixed so results are reproducible. Changing it invalidates every recorded eval run.
SEED = 20260921

#: All generated dates are relative to this, never to ``date.today()``.
EPOCH = date(2025, 1, 1)
DAYS_SPAN = 540

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE customers (
    customer_id   INTEGER PRIMARY KEY,
    name_en       TEXT NOT NULL,
    name_ar       TEXT NOT NULL,
    city          TEXT NOT NULL,
    country       TEXT NOT NULL,
    segment       TEXT NOT NULL,
    signed_up_on  TEXT NOT NULL
);

CREATE TABLE warehouses (
    warehouse_id  INTEGER PRIMARY KEY,
    name_en       TEXT NOT NULL,
    name_ar       TEXT NOT NULL,
    city          TEXT NOT NULL,
    country       TEXT NOT NULL,
    capacity_m3   INTEGER NOT NULL
);

CREATE TABLE couriers (
    courier_id    INTEGER PRIMARY KEY,
    name_en       TEXT NOT NULL,
    name_ar       TEXT NOT NULL,
    country       TEXT NOT NULL,
    is_active     INTEGER NOT NULL
);

CREATE TABLE products (
    product_id    INTEGER PRIMARY KEY,
    name_en       TEXT NOT NULL,
    name_ar       TEXT NOT NULL,
    category      TEXT NOT NULL,
    unit_price_aed REAL NOT NULL,
    weight_kg     REAL NOT NULL
);

CREATE TABLE orders (
    order_id      INTEGER PRIMARY KEY,
    customer_id   INTEGER NOT NULL,
    warehouse_id  INTEGER NOT NULL,
    courier_id    INTEGER NOT NULL,
    placed_at     TEXT NOT NULL,
    promised_at   TEXT NOT NULL,
    delivered_at  TEXT,
    status        TEXT NOT NULL,
    total_aed     REAL NOT NULL,
    FOREIGN KEY (customer_id)  REFERENCES customers(customer_id),
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id),
    FOREIGN KEY (courier_id)   REFERENCES couriers(courier_id)
);

CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL,
    product_id    INTEGER NOT NULL,
    quantity      INTEGER NOT NULL,
    unit_price_aed REAL NOT NULL,
    FOREIGN KEY (order_id)   REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_status   ON orders(status);
CREATE INDEX idx_items_order     ON order_items(order_id);
"""

CITIES: tuple[tuple[str, str, str], ...] = (
    ("Dubai", "دبي", "UAE"),
    ("Abu Dhabi", "أبو ظبي", "UAE"),
    ("Sharjah", "الشارقة", "UAE"),
    ("Riyadh", "الرياض", "Saudi Arabia"),
    ("Jeddah", "جدة", "Saudi Arabia"),
    ("Dammam", "الدمام", "Saudi Arabia"),
    ("Doha", "الدوحة", "Qatar"),
    ("Kuwait City", "مدينة الكويت", "Kuwait"),
    ("Manama", "المنامة", "Bahrain"),
    ("Muscat", "مسقط", "Oman"),
)

FIRST_NAMES: tuple[tuple[str, str], ...] = (
    ("Ahmed", "أحمد"), ("Fatima", "فاطمة"), ("Omar", "عمر"), ("Layla", "ليلى"),
    ("Yousef", "يوسف"), ("Noura", "نورة"), ("Khalid", "خالد"), ("Maryam", "مريم"),
    ("Saeed", "سعيد"), ("Hessa", "حصة"), ("Rashid", "راشد"), ("Amal", "أمل"),
    ("Tariq", "طارق"), ("Salma", "سلمى"), ("Bilal", "بلال"), ("Huda", "هدى"),
)

LAST_NAMES: tuple[tuple[str, str], ...] = (
    ("Al Mansouri", "المنصوري"), ("Al Farsi", "الفارسي"), ("Al Qassimi", "القاسمي"),
    ("Al Harbi", "الحربي"), ("Al Otaibi", "العتيبي"), ("Al Thani", "آل ثاني"),
    ("Al Balushi", "البلوشي"), ("Al Sabah", "الصباح"), ("Al Khalifa", "آل خليفة"),
)

SEGMENTS = ("enterprise", "sme", "retail")
ORDER_STATUS = ("pending", "in_transit", "delivered", "returned", "cancelled")

PRODUCTS: tuple[tuple[str, str, str, float, float], ...] = (
    ("Dates Gift Box 1kg", "علبة تمر هدية ١ كجم", "food", 120.0, 1.2),
    ("Arabic Coffee Beans 500g", "بن عربي ٥٠٠ جرام", "food", 65.0, 0.55),
    ("Cardamom 250g", "هيل ٢٥٠ جرام", "food", 48.0, 0.28),
    ("Oud Perfume 50ml", "عطر عود ٥٠ مل", "fragrance", 890.0, 0.4),
    ("Bakhoor Set", "طقم بخور", "fragrance", 210.0, 0.9),
    ("Prayer Rug", "سجادة صلاة", "home", 150.0, 1.8),
    ("Majlis Cushion", "مسند مجلس", "home", 240.0, 3.2),
    ("Arabic Calligraphy Frame", "لوحة خط عربي", "home", 480.0, 2.5),
    ("Camel Milk Chocolate", "شوكولاتة حليب الإبل", "food", 55.0, 0.3),
    ("Saffron 5g", "زعفران ٥ جرام", "food", 320.0, 0.01),
    ("Rose Water 300ml", "ماء ورد ٣٠٠ مل", "food", 32.0, 0.35),
    ("Ceramic Dallah", "دلة سيراميك", "home", 175.0, 1.4),
)

COURIERS: tuple[tuple[str, str, str], ...] = (
    ("Gulf Express", "الخليج السريع", "UAE"),
    ("Desert Logistics", "لوجستيات الصحراء", "Saudi Arabia"),
    ("Pearl Delivery", "توصيل اللؤلؤة", "Qatar"),
    ("Sand Road Freight", "شحن طريق الرمال", "UAE"),
    ("Falcon Courier", "الصقر للتوصيل", "Saudi Arabia"),
)


def _iso(day_offset: int, rng: random.Random) -> str:
    """A timestamp ``day_offset`` days after :data:`EPOCH`, with a seeded time of day."""
    stamp = datetime.combine(EPOCH + timedelta(days=day_offset), datetime.min.time())
    stamp += timedelta(hours=rng.randrange(6, 22), minutes=rng.randrange(0, 60))
    return stamp.strftime("%Y-%m-%d %H:%M:%S")


def build(
    path: Path,
    *,
    n_customers: int = 180,
    n_orders: int = 900,
    seed: int = SEED,
    overwrite: bool = True,
) -> Path:
    """Create the demo database at ``path``. Returns the path for convenience."""
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not overwrite:
            logger.info("database exists, leaving it alone", extra={"path": str(path)})
            return path
        path.unlink()

    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA_SQL)

        warehouses = [
            (
                i + 1,
                f"{city_en} Hub",
                f"مستودع {city_ar}",
                city_en,
                country,
                rng.randrange(4_000, 40_000, 500),
            )
            for i, (city_en, city_ar, country) in enumerate(CITIES[:6])
        ]
        conn.executemany("INSERT INTO warehouses VALUES (?,?,?,?,?,?)", warehouses)

        couriers = [
            (i + 1, en, ar, country, 1 if i != len(COURIERS) - 1 else 0)
            for i, (en, ar, country) in enumerate(COURIERS)
        ]
        conn.executemany("INSERT INTO couriers VALUES (?,?,?,?,?)", couriers)

        products = [
            (i + 1, en, ar, cat, price, weight)
            for i, (en, ar, cat, price, weight) in enumerate(PRODUCTS)
        ]
        conn.executemany("INSERT INTO products VALUES (?,?,?,?,?,?)", products)

        customers = []
        for cid in range(1, n_customers + 1):
            first_en, first_ar = rng.choice(FIRST_NAMES)
            last_en, last_ar = rng.choice(LAST_NAMES)
            city_en, _city_ar, country = rng.choice(CITIES)
            customers.append(
                (
                    cid,
                    f"{first_en} {last_en}",
                    f"{first_ar} {last_ar}",
                    city_en,
                    country,
                    rng.choices(SEGMENTS, weights=(1, 3, 6))[0],
                    (EPOCH + timedelta(days=rng.randrange(0, 300))).isoformat(),
                )
            )
        conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?,?)", customers)

        orders: list[tuple[object, ...]] = []
        items: list[tuple[object, ...]] = []
        item_id = 1
        for oid in range(1, n_orders + 1):
            customer_id = rng.randrange(1, n_customers + 1)
            placed_offset = rng.randrange(0, DAYS_SPAN)
            promised_offset = placed_offset + rng.randrange(2, 9)
            status = rng.choices(ORDER_STATUS, weights=(8, 12, 68, 7, 5))[0]

            delivered_at: str | None = None
            if status == "delivered":
                # ~22% of delivered orders land after the promised date. This is the whole
                # point of the schema: "late" is a computed relationship between two
                # columns, not a stored flag, so the model has to express real logic.
                if rng.random() < 0.22:
                    delivered_offset = promised_offset + rng.randrange(1, 12)
                else:
                    delivered_offset = placed_offset + rng.randrange(
                        1, promised_offset - placed_offset + 1
                    )
                delivered_at = _iso(delivered_offset, rng)

            n_lines = rng.choices((1, 2, 3, 4), weights=(50, 30, 15, 5))[0]
            chosen = rng.sample(range(1, len(PRODUCTS) + 1), n_lines)
            total = 0.0
            for product_id in chosen:
                quantity = rng.choices((1, 2, 3, 5, 10), weights=(55, 22, 12, 7, 4))[0]
                unit_price = PRODUCTS[product_id - 1][3]
                # A seeded discount keeps line price distinct from catalogue price, so a
                # query that naively joins to products.unit_price_aed gets a different
                # answer than one that reads order_items.unit_price_aed. That distinction
                # is exactly what separates a correct query from a plausible one.
                unit_price = round(unit_price * rng.choice((1.0, 1.0, 0.9, 0.85)), 2)
                items.append((item_id, oid, product_id, quantity, unit_price))
                total += quantity * unit_price
                item_id += 1

            orders.append(
                (
                    oid,
                    customer_id,
                    rng.randrange(1, len(warehouses) + 1),
                    rng.randrange(1, len(COURIERS) + 1),
                    _iso(placed_offset, rng),
                    _iso(promised_offset, rng),
                    delivered_at,
                    status,
                    round(total, 2),
                )
            )

        conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?)", orders)
        conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", items)
        conn.commit()

        logger.info(
            "synthetic database built",
            extra={
                "path": str(path),
                "customers": len(customers),
                "orders": len(orders),
                "order_items": len(items),
            },
        )
        return path
    finally:
        conn.close()


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    from ..config import Settings

    settings = Settings.from_env()
    build(settings.db_path)
    print(f"built {settings.db_path}")
