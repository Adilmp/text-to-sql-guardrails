"""The curated question list: what the cache is pre-filled with and what suggestions offer.

One list, used in two places. ``scripts/prefill_cache.py`` asks the model every question here
so each one answers from the cache; the server's ``/api/suggest`` offers them as the user
types, marked "instant" when they are cached.

Suggestions come from this list only, never from what other people asked. Questions asked on
the server are cached too, but offering them as suggestions would show one user's questions
to the next ("how many orders did Ahmed Al Harbi cancel?"), which is a privacy leak the
moment there is more than one user (DECISIONS.md D41).

In order: the demo page's examples, every eval question (English, Arabic, Urdu), general
questions about the whole database, then templated questions for every city, country, order
status, product category, customer segment, courier, warehouse and product.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .cache import canonical_question
from .eval import build_holdout, build_suite
from .nl import arabic_ratio, detect_script

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


def curated_questions(catalog_names: dict[str, list[tuple[str, str]]]) -> list[tuple[str, str]]:
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


def names_from_db(db_path: Path) -> dict[str, list[tuple[str, str]]]:
    """English and Arabic names straight from the database, so templates can't drift."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
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


@dataclass(frozen=True)
class Suggestion:
    question: str
    #: Cached for the current prompt and model: picking it answers in milliseconds.
    instant: bool

    def to_dict(self) -> dict[str, object]:
        return {"question": self.question, "instant": self.instant}


class Suggester:
    """Curated questions that match what the user has typed so far.

    Matching works on the cache's canonical form, so it ignores case, diacritics, Arabic and
    Urdu letter variants and digit scripts. Every typed word has to start a word of the
    question; an Arabic-script word of three letters or more may also sit inside one, so
    "طلبات" finds "الطلبات" (the article is written attached). English words don't get that
    second rule, or "late" would find "chocolate". Cached questions come first, because
    choosing one answers instantly, then questions in the language being typed.
    """

    def __init__(self, questions: Iterable[str]) -> None:
        self._entries: list[tuple[str, str, list[str], str]] = []
        for question in questions:
            key = canonical_question(question)
            self._entries.append((question, key, key.split(), detect_script(question).value))

    def __len__(self) -> int:
        return len(self._entries)

    def suggest(
        self, text: str, *, is_cached: Callable[[str], bool], limit: int = 8
    ) -> list[Suggestion]:
        typed = canonical_question(text).split()
        if not typed:
            return []
        prefix = " ".join(typed)
        language = detect_script(text).value
        ranked: list[tuple[bool, bool, bool, int, str, bool]] = []
        for question, key, tokens, question_language in self._entries:
            if all(_word_matches(word, tokens) for word in typed):
                instant = is_cached(key)
                # Sort keys: instant first, then the language being typed, then questions
                # that start with what was typed, then shorter ones.
                ranked.append(
                    (
                        not instant,
                        question_language != language,
                        not key.startswith(prefix),
                        len(key),
                        question,
                        instant,
                    )
                )
        ranked.sort()
        return [Suggestion(question, instant) for *_, question, instant in ranked[:limit]]


def _word_matches(word: str, tokens: list[str]) -> bool:
    inside = len(word) >= 3 and arabic_ratio(word) > 0.5
    return any(t.startswith(word) or (inside and word in t) for t in tokens)
