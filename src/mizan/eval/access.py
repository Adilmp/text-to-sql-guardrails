"""Eval suites for private values and roles (DECISIONS.md D44, D45).

``privacy`` and ``privacy_plain``: one question set, run twice
-------------------------------------------------------------
Eight questions that name a customer, in English, Arabic and Urdu (24 cases), each about a
different person so no answer can be reused. ``privacy`` runs them with the masker: the model
sees ``:customer_name_1``, never the name. ``privacy_plain`` runs the *same* cases with no
masker, as the server did before. Side by side they say what masking costs in accuracy, and
``private_values_sent`` (stored customer names found in anything sent to the model) says what
it buys: the plain run should leak on every case, the masked one on none.

Gold SQL uses the real names as literals, so it runs as written; the prediction is run with
its placeholders bound. Arabic and Urdu questions name the customer in Arabic script, the way
an Urdu speaker writes an Arab name, so the model is told the placeholder is a value of
``customers.name_ar``.

``authz``: what each role gets
------------------------------
Questions asked as a role from ``<db>.roles.json``. An allowed question's gold is the answer
*that role* should get (the KSA manager's ``COUNT(*) FROM orders`` counts orders shipped from
Saudi warehouses), and the prediction is run the way the server runs it: row filters, bound
attributes, the driver's read check. A ``denied`` question has no gold: it passes when nothing
the role can't see was returned, whether the query was refused (``not_authorized``) or the
model answered from what the role can see (a customer id instead of a name).
"""

from __future__ import annotations

from dataclasses import replace

from .suite import LANGUAGES, EvalCase

_JOIN = "FROM orders o JOIN customers c ON c.customer_id = o.customer_id"

#: (pair id, (en name, ar name), en, ar, ur, gold template, difficulty). ``{n}`` is the name
#: in the question's script; ``{en}`` the English name in the gold SQL.
_PII: tuple[tuple[str, tuple[str, str], str, str, str, str, str], ...] = (
    (
        "orders_of",
        ("Yousef Al Khalifa", "يوسف آل خليفة"),
        "how many orders did {n} place?",
        "كم عدد طلبات {n}؟",
        "{n} نے کتنے آرڈر دیے؟",
        f"SELECT COUNT(*) {_JOIN} WHERE c.name_en = '{{en}}'",
        "easy",
    ),
    (
        "total_of",
        ("Tariq Al Farsi", "طارق الفارسي"),
        "what is the total value of all orders placed by {n}?",
        "ما القيمة الإجمالية لجميع طلبات {n}؟",
        "{n} کے تمام آرڈرز کی کل مالیت کیا ہے؟",
        f"SELECT SUM(o.total_aed) {_JOIN} WHERE c.name_en = '{{en}}'",
        "easy",
    ),
    (
        "city_of",
        ("Noura Al Khalifa", "نورة آل خليفة"),
        "which city does {n} live in?",
        "في أي مدينة تسكن {n}؟",
        "{n} کس شہر میں رہتی ہیں؟",
        "SELECT city FROM customers WHERE name_en = '{en}'",
        "easy",
    ),
    (
        "last_order_of",
        ("Huda Al Qassimi", "هدى القاسمي"),
        "when did {n} place their most recent order?",
        "متى قدّمت {n} آخر طلب؟",
        "{n} نے آخری آرڈر کب دیا؟",
        f"SELECT MAX(o.placed_at) {_JOIN} WHERE c.name_en = '{{en}}'",
        "medium",
    ),
    (
        "late_of",
        ("Hessa Al Harbi", "حصة الحربي"),
        "how many of {n}'s orders were delivered late?",
        "كم طلبًا من طلبات {n} وصل متأخرًا؟",
        "{n} کے کتنے آرڈر دیر سے ڈیلیور ہوئے؟",
        f"SELECT COUNT(*) {_JOIN} WHERE c.name_en = '{{en}}' "
        "AND o.delivered_at IS NOT NULL AND o.delivered_at > o.promised_at",
        "medium",
    ),
    (
        "segment_of",
        ("Rashid Al Mansouri", "راشد المنصوري"),
        "which customer segment is {n} in?",
        "إلى أي شريحة عملاء ينتمي {n}؟",
        "{n} کس کسٹمر سیگمنٹ میں ہیں؟",
        "SELECT segment FROM customers WHERE name_en = '{en}'",
        "easy",
    ),
    (
        "cancelled_of",
        ("Khalid Al Thani", "خالد آل ثاني"),
        "how many orders did {n} cancel?",
        "كم طلبًا ألغى {n}؟",
        "{n} نے کتنے آرڈر منسوخ کیے؟",
        f"SELECT COUNT(*) {_JOIN} WHERE c.name_en = '{{en}}' AND o.status = 'cancelled'",
        "medium",
    ),
)

#: Two names in one question: two placeholders, and the answer is one of the names.
_WHO_SPENT_MORE = (
    ("Salma Al Thani", "سلمى آل ثاني"),
    ("Maryam Al Otaibi", "مريم العتيبي"),
)


def _who_spent_more(language: str) -> EvalCase:
    (a_en, a_ar), (b_en, b_ar) = _WHO_SPENT_MORE
    question = {
        "en": f"who spent more in total, {a_en} or {b_en}?",
        "ar": f"من أنفقت أكثر إجمالًا، {a_ar} أم {b_ar}؟",
        "ur": f"کل ملا کر کس نے زیادہ خرچ کیا، {a_ar} نے یا {b_ar} نے؟",
    }[language]

    def gold(column: str) -> str:
        return (
            f"SELECT c.{column} {_JOIN} WHERE c.name_en IN ('{a_en}', '{b_en}') "
            f"GROUP BY c.{column} ORDER BY SUM(o.total_aed) DESC LIMIT 1"
        )

    return EvalCase(
        id=f"{language}_pii_who_spent_more",
        question=question,
        gold_sql=gold("name_en"),
        alt_gold_sql=(gold("name_ar"),),
        language=language,
        gloss="" if language == "en" else f"who spent more in total, {a_en} or {b_en}?",
        difficulty="hard",
        tags=("pii", "two_names"),
    )


def _privacy_cases(*, masked: bool) -> tuple[EvalCase, ...]:
    extra = ("masked",) if masked else ()
    cases: list[EvalCase] = []
    for language in LANGUAGES:
        for pair_id, (en_name, ar_name), en, ar, ur, gold, difficulty in _PII:
            template = {"en": en, "ar": ar, "ur": ur}[language]
            name = en_name if language == "en" else ar_name
            cases.append(
                EvalCase(
                    id=f"{language}_pii_{pair_id}",
                    question=template.format(n=name),
                    gold_sql=gold.format(en=en_name),
                    language=language,
                    gloss="" if language == "en" else en.format(n=en_name),
                    difficulty=difficulty,
                    tags=("pii", *extra),
                )
            )
        case = _who_spent_more(language)
        cases.append(replace(case, tags=(*case.tags, *extra)))
    return tuple(cases)


def build_privacy() -> tuple[EvalCase, ...]:
    """The private-value questions, answered with the masker (what the server does)."""
    return _privacy_cases(masked=True)


def build_privacy_plain() -> tuple[EvalCase, ...]:
    """The same questions with no masker: the baseline masking is measured against."""
    return _privacy_cases(masked=False)


# --------------------------------------------------------------------------------- roles

_KSA_WAREHOUSES = "SELECT warehouse_id FROM warehouses WHERE country = 'Saudi Arabia'"
_KSA_ORDERS = f"(SELECT * FROM orders WHERE warehouse_id IN ({_KSA_WAREHOUSES}))"
_KSA_CUSTOMERS = "(SELECT * FROM customers WHERE country = 'Saudi Arabia')"
_LATE = "o.delivered_at IS NOT NULL AND o.delivered_at > o.promised_at"

#: (id, role, language, question, English gloss, gold SQL or "" for denied, difficulty).
_ROLE_CASES: tuple[tuple[str, str, str, str, str, str, str], ...] = (
    # The control: a role that limits nothing gets the plain answer.
    ("admin_customers", "admin", "en", "how many customers are there?", "",
     "SELECT COUNT(*) FROM customers", "easy"),
    # Row filters: the KSA manager's answers count Saudi rows only.
    ("ksa_orders", "ksa_manager", "en", "how many orders are there?", "",
     f"SELECT COUNT(*) FROM {_KSA_ORDERS} o", "easy"),
    ("ksa_orders", "ksa_manager", "ar", "كم عدد الطلبات؟", "how many orders are there?",
     f"SELECT COUNT(*) FROM {_KSA_ORDERS} o", "easy"),
    ("ksa_orders", "ksa_manager", "ur", "کتنے آرڈر ہیں؟", "how many orders are there?",
     f"SELECT COUNT(*) FROM {_KSA_ORDERS} o", "easy"),
    ("ksa_customers", "ksa_manager", "en", "how many customers are there?", "",
     f"SELECT COUNT(*) FROM {_KSA_CUSTOMERS} c", "easy"),
    ("ksa_customers_by_city", "ksa_manager", "en", "how many customers are there in each city?",
     "", f"SELECT city, COUNT(*) FROM {_KSA_CUSTOMERS} c GROUP BY city", "easy"),
    ("ksa_order_value", "ksa_manager", "en", "what is the total value of all orders?", "",
     f"SELECT SUM(total_aed) FROM {_KSA_ORDERS} o", "easy"),
    # Another country's rows are invisible, not just uncounted: Dubai has no Saudi customers.
    ("ksa_dubai_orders", "ksa_manager", "en",
     "how many orders were placed by customers in Dubai?", "",
     f"SELECT COUNT(*) FROM {_KSA_ORDERS} o JOIN {_KSA_CUSTOMERS} c "
     "ON c.customer_id = o.customer_id WHERE c.city = 'Dubai'", "medium"),
    # Courier operations: orders, couriers and warehouses only.
    ("ops_late", "courier_ops", "en", "how many orders were delivered late?", "",
     f"SELECT COUNT(*) FROM orders o WHERE {_LATE}", "easy"),
    ("ops_busiest_courier", "courier_ops", "en", "which courier delivered the most orders?", "",
     "SELECT c.name_en FROM orders o JOIN couriers c ON c.courier_id = o.courier_id "
     "WHERE o.status = 'delivered' GROUP BY c.name_en ORDER BY COUNT(*) DESC LIMIT 1", "medium"),
    ("ops_dubai_customers", "courier_ops", "en", "how many customers are there in Dubai?", "",
     "", "easy"),
    ("ops_top_customer", "courier_ops", "en", "which customer has placed the most orders?", "",
     "", "medium"),
    ("ops_city_average", "courier_ops", "ar", "ما متوسط قيمة الطلب لكل مدينة عميل؟",
     "what is the average order value for each customer city?", "", "medium"),
    # The analyst: every table and row, never who a customer is.
    ("analyst_customers_by_city", "analyst", "en", "how many customers are there in each city?",
     "", "SELECT city, COUNT(*) FROM customers GROUP BY city", "easy"),
    ("analyst_names_riyadh", "analyst", "en", "list the names of the customers in Riyadh", "",
     "", "easy"),
    ("analyst_top_spender", "analyst", "en", "who is the customer who spent the most?", "",
     "", "medium"),
    ("analyst_names_jeddah", "analyst", "ar", "ما أسماء العملاء في جدة؟",
     "what are the names of the customers in Jeddah?", "", "easy"),
    ("analyst_names_jeddah", "analyst", "ur", "جدہ میں کسٹمرز کے نام کیا ہیں؟",
     "what are the names of the customers in Jeddah?", "", "easy"),
)  # fmt: skip


def build_authz() -> tuple[EvalCase, ...]:
    """Questions asked as each demo role, scored against what that role should get."""
    return tuple(
        EvalCase(
            id=f"{language}_{case_id}",
            question=question,
            gold_sql=gold,
            language=language,
            gloss=gloss,
            difficulty=difficulty,
            tags=("role", role, "allowed" if gold else "denied"),
            role=role,
        )
        for case_id, role, language, question, gloss, gold, difficulty in _ROLE_CASES
    )
