#!/usr/bin/env python
"""Pre-fill the server's answer cache with the questions people are likely to ask.

Every question here goes through the model once, exactly as if a user had asked it, and is
stored only if the cache would store it anyway (confident, no repair, every filtered value
found in the data). Afterwards those questions answer in milliseconds instead of seconds.

What is in the list, in the order it is asked:

1. the demo page's example questions;
2. every question in the development and held-out eval suites (English, Arabic, Urdu);
3. general questions about the whole database, in the three languages;
4. templated questions for every city, country, order status, product category, customer
   segment, courier, warehouse and product, in the three languages.

The answers are the model's own, mistakes included. The eval suites' gold queries are known,
but planting them would make the demo look more accurate than the model is, so they are
never used here.

The run is resumable: questions already cached are skipped, so stopping and starting again
loses nothing. The server can stay up meanwhile; a visitor's question waits at most for the
one question being pre-filled (Ollama answers one request at a time).

Usage:
    python scripts/prefill_cache.py                  # everything (~450 questions, ~1 h on a CPU)
    python scripts/prefill_cache.py --only examples suite
    python scripts/prefill_cache.py --dry-run        # list the questions, ask nothing
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mizan.api import serving_settings  # noqa: E402
from mizan.cache import (  # noqa: E402
    AnswerCache,
    CachedTextToSQL,
    CacheSettings,
    cacheable,
    canonical_question,
)
from mizan.config import Settings  # noqa: E402
from mizan.eval import build_holdout, build_suite  # noqa: E402
from mizan.generate import TextToSQL  # noqa: E402
from mizan.logging import configure  # noqa: E402
from mizan.providers import build_provider  # noqa: E402
from mizan.schema import load_catalog  # noqa: E402

#: The demo page's example buttons (web/index.html, EXAMPLES).
EXAMPLES = (
    "how many orders were delivered late?",
    "which courier delivered late most often?",
    "كم عدد الطلبات المتأخرة في دبي؟",
    "ما متوسط قيمة الطلب لكل مدينة؟",
    "کتنے آرڈر ابھی تک ڈیلیور نہیں ہوئے؟",
)

#: (English, Arabic, Urdu) questions about the database as a whole.
GENERAL = (
    ("what is the total revenue?", "ما إجمالي الإيرادات؟", "کل آمدنی کتنی ہے؟"),
    ("what is the average order value?", "ما متوسط قيمة الطلب؟", "آرڈر کی اوسط مالیت کتنی ہے؟"),
    ("how many products are there?", "كم عدد المنتجات؟", "کتنی پروڈکٹس ہیں؟"),
    ("how many couriers are there?", "كم عدد شركات التوصيل؟", "کتنے کورئیر ہیں؟"),
    ("how many warehouses are there?", "كم عدد المستودعات؟", "کتنے گودام ہیں؟"),
    ("how many customers are there?", "كم عدد العملاء؟", "کتنے کسٹمرز ہیں؟"),
    (
        "which city has the most customers?",
        "ما المدينة التي فيها أكبر عدد من العملاء؟",
        "کس شہر میں سب سے زیادہ کسٹمرز ہیں؟",
    ),
    (
        "which warehouse fulfilled the most orders?",
        "أي مستودع نفّذ أكبر عدد من الطلبات؟",
        "کس گودام نے سب سے زیادہ آرڈر پورے کیے؟",
    ),
    (
        "which product earned the most revenue?",
        "ما المنتج الذي حقق أعلى إيرادات؟",
        "کس پروڈکٹ نے سب سے زیادہ آمدنی کمائی؟",
    ),
    (
        "which customer placed the most orders?",
        "أي عميل قدّم أكبر عدد من الطلبات؟",
        "کس کسٹمر نے سب سے زیادہ آرڈر دیے؟",
    ),
    (
        "how many orders were placed in 2026?",
        "كم عدد الطلبات التي قُدّمت في عام ٢٠٢٦؟",
        "۲۰۲۶ میں کتنے آرڈر دیے گئے؟",
    ),
    (
        "what is the total revenue for each country?",
        "ما إجمالي الإيرادات لكل دولة؟",
        "ہر ملک کی کل آمدنی کتنی ہے؟",
    ),
)

#: (English value, Arabic name, Urdu name). English is how the value is stored.
CITIES = (
    ("Abu Dhabi", "أبوظبي", "ابوظہبی"),
    ("Dammam", "الدمام", "دمام"),
    ("Doha", "الدوحة", "دوحہ"),
    ("Dubai", "دبي", "دبئی"),
    ("Jeddah", "جدة", "جدہ"),
    ("Kuwait City", "مدينة الكويت", "کویت سٹی"),
    ("Manama", "المنامة", "منامہ"),
    ("Muscat", "مسقط", "مسقط"),
    ("Riyadh", "الرياض", "ریاض"),
    ("Sharjah", "الشارقة", "شارجہ"),
)
COUNTRIES = (
    ("Bahrain", "البحرين", "بحرین"),
    ("Kuwait", "الكويت", "کویت"),
    ("Oman", "عمان", "عمان"),
    ("Qatar", "قطر", "قطر"),
    ("Saudi Arabia", "السعودية", "سعودی عرب"),
    ("the UAE", "الإمارات", "متحدہ عرب امارات"),
)
#: Status questions differ in grammar per status, so each is a full sentence.
STATUSES = (
    ("how many orders are delivered?", "كم عدد الطلبات المسلّمة؟", "کتنے آرڈر ڈیلیور ہو چکے ہیں؟"),
    ("how many orders are pending?", "كم عدد الطلبات قيد الانتظار؟", "کتنے آرڈر زیر التوا ہیں؟"),
    ("how many orders are in transit?", "كم عدد الطلبات قيد الشحن؟", "کتنے آرڈر راستے میں ہیں؟"),
    ("how many orders were cancelled?", "كم عدد الطلبات الملغاة؟", "کتنے آرڈر منسوخ ہوئے؟"),
    ("how many orders were returned?", "كم عدد الطلبات المرتجعة؟", "کتنے آرڈر واپس ہوئے؟"),
)
CATEGORIES = (
    ("food", "الأغذية", "کھانے پینے کی"),
    ("fragrance", "العطور", "خوشبو کی"),
    ("home", "المنزل", "گھریلو"),
)
SEGMENTS = (
    ("enterprise", "الشركات", "انٹرپرائز"),
    ("sme", "الشركات الصغيرة والمتوسطة", "ایس ایم ای"),
    ("retail", "التجزئة", "ریٹیل"),
)


def templated(catalog_names: dict[str, list[tuple[str, str]]]) -> list[tuple[str, str]]:
    """``(group, question)`` for every template and value, in English, Arabic and Urdu."""
    out: list[tuple[str, str]] = []

    def add(group: str, *questions: str) -> None:
        out.extend((group, q) for q in questions)

    for en, ar, ur in CITIES:
        add(
            "city",
            f"how many customers are in {en}?",
            f"كم عدد العملاء في {ar}؟",
            f"{ur} میں کتنے کسٹمرز ہیں؟",
            f"how many orders came from customers in {en}?",
            f"كم عدد الطلبات من العملاء في {ar}؟",
            f"{ur} کے کسٹمرز نے کتنے آرڈر دیے؟",
            f"how many orders to customers in {en} were delivered late?",
            f"كم عدد طلبات العملاء في {ar} التي تم تسليمها متأخرة؟",
            f"{ur} کے کسٹمرز کے کتنے آرڈر تاخیر سے ڈیلیور ہوئے؟",
            f"what is the average order value for customers in {en}?",
            f"ما متوسط قيمة الطلب للعملاء في {ar}؟",
            f"{ur} کے کسٹمرز کے آرڈر کی اوسط مالیت کتنی ہے؟",
        )
    for en, ar, ur in COUNTRIES:
        add(
            "country",
            f"how many customers are in {en}?",
            f"كم عدد العملاء في {ar}؟",
            f"{ur} میں کتنے کسٹمرز ہیں؟",
            f"how many orders came from customers in {en}?",
            f"كم عدد الطلبات من العملاء في {ar}؟",
            f"{ur} کے کسٹمرز نے کتنے آرڈر دیے؟",
        )
    for questions in STATUSES:
        add("status", *questions)
    for en, ar, ur in CATEGORIES:
        add(
            "category",
            f"what is the total revenue from {en} products?",
            f"ما إجمالي الإيرادات من منتجات {ar}؟",
            f"{ur} پروڈکٹس سے کل آمدنی کتنی ہے؟",
            f"which products are in the {en} category?",
            f"ما المنتجات في فئة {ar}؟",
            f"{ur} کیٹیگری میں کون سی پروڈکٹس ہیں؟",
        )
    for en, ar, ur in SEGMENTS:
        add(
            "segment",
            f"how many {en} customers are there?",
            f"كم عدد عملاء فئة {ar}؟",
            f"{ur} کے کتنے کسٹمرز ہیں؟",
            f"what is the average order value for {en} customers?",
            f"ما متوسط قيمة الطلب لعملاء فئة {ar}؟",
            f"{ur} کسٹمرز کے آرڈر کی اوسط مالیت کتنی ہے؟",
        )
    # Names in Arabic come from the database's own name_ar column; Urdu speakers write brand
    # and product names in English, so the Urdu questions keep the English name.
    for en, ar in catalog_names["couriers"]:
        add(
            "courier",
            f"how many orders did {en} handle?",
            f"كم عدد الطلبات التي تولتها {ar}؟",
            f"{en} نے کتنے آرڈر سنبھالے؟",
            f"how many orders did {en} deliver late?",
            f"كم عدد الطلبات التي سلمتها {ar} متأخرة؟",
            f"{en} نے کتنے آرڈر تاخیر سے ڈیلیور کیے؟",
        )
    for en, ar in catalog_names["warehouses"]:
        add(
            "warehouse",
            f"how many orders did {en} fulfil?",
            f"كم عدد الطلبات التي نفذها {ar}؟",
            f"{en} نے کتنے آرڈر پورے کیے؟",
        )
    for en, ar in catalog_names["products"]:
        add(
            "product",
            f"how many units of {en} were sold?",
            f"كم عدد الوحدات المباعة من {ar}؟",
            f"{en} کے کتنے یونٹس فروخت ہوئے؟",
            f"what is the total revenue from {en}?",
            f"ما إجمالي الإيرادات من {ar}؟",
            f"{en} سے کل آمدنی کتنی ہے؟",
        )
    return out


def all_questions(catalog_names: dict[str, list[tuple[str, str]]]) -> list[tuple[str, str]]:
    """Every question to pre-fill, most likely first, without repeats."""
    ordered: list[tuple[str, str]] = [("examples", q) for q in EXAMPLES]
    ordered += [("suite", c.question) for c in (*build_suite(), *build_holdout())]
    ordered += [("general", q) for triple in GENERAL for q in triple]
    ordered += templated(catalog_names)
    seen: set[str] = set()
    unique = []
    for group, question in ordered:
        key = canonical_question(question)
        if key not in seen:
            seen.add(key)
            unique.append((group, question))
    return unique


def _names(settings: Settings) -> dict[str, list[tuple[str, str]]]:
    """English and Arabic names straight from the database, so templates can't drift."""
    import sqlite3

    conn = sqlite3.connect(f"file:{settings.db_path}?mode=ro", uri=True)
    try:
        return {
            table: [
                (str(r[0]), str(r[1]))
                for r in conn.execute(
                    f"SELECT name_en, name_ar FROM {table} ORDER BY rowid"  # noqa: S608 - fixed names
                )
            ]
            for table in ("couriers", "warehouses", "products")
        }
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    groups = (
        "examples",
        "suite",
        "general",
        "city",
        "country",
        "status",
        "category",
        "segment",
        "courier",
        "warehouse",
        "product",
    )
    parser.add_argument("--only", nargs="+", choices=groups, help="pre-fill only these groups")
    parser.add_argument("--limit", type=int, help="stop after this many model calls")
    parser.add_argument("--dry-run", action="store_true", help="list the questions and exit")
    args = parser.parse_args()

    # The same settings the server uses, so entries land in the server's cache context and
    # the model's keep-alive matches (a different keep-alive would change how long Ollama
    # keeps the model loaded after this script's last request).
    settings = serving_settings(Settings.from_env())
    questions = all_questions(_names(settings))
    if args.only:
        questions = [(g, q) for g, q in questions if g in args.only]
    if args.dry_run:
        for group, question in questions:
            print(f"{group:10} {question}")
        print(f"\n{len(questions)} questions")
        return 0

    configure(settings.log_level, settings.log_dir, console=False)
    cache_settings = CacheSettings.from_env()
    if not cache_settings.enabled:
        print("MIZAN_CACHE is off: nothing to pre-fill", file=sys.stderr)
        return 1
    catalog = load_catalog(settings.db_path)
    engine = TextToSQL(catalog, build_provider(settings), settings)
    cache = AnswerCache(cache_settings.path)
    served = CachedTextToSQL(engine, cache, cache_settings)

    outcome: Counter[str] = Counter()
    asked = 0
    started = time.perf_counter()
    for index, (group, question) in enumerate(questions, start=1):
        if cache.has(served.context, canonical_question(question)):
            outcome["already cached"] += 1
            continue
        if args.limit is not None and asked >= args.limit:
            break
        asked += 1
        answer, _ = served.ask(question)
        if cacheable(answer, cache_settings.min_confidence):
            verdict = "cached"
        elif not answer.ok:
            verdict = "not cached: no usable answer"
        elif answer.repairs:
            verdict = "not cached: needed a repair"
        else:
            verdict = "not cached: below the confidence bar"
        outcome[verdict] += 1
        print(
            f"[{index}/{len(questions)}] {verdict:38} {answer.latency_ms / 1000:5.1f}s "
            f"{group:9} {question}",
            flush=True,
        )

    minutes = (time.perf_counter() - started) / 60
    print(
        f"\n{asked} asked in {minutes:.1f} min; cache now holds "
        f"{cache.count(served.context)} answers for this prompt and model"
    )
    for verdict, count in outcome.most_common():
        print(f"  {verdict}: {count}")
    cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
