from __future__ import annotations

import os
import re
from dataclasses import dataclass

import pandas as pd

from finance.analytics import (
    add_expense_columns,
    build_kpis,
    category_breakdown,
    detect_recurring_expenses,
    financial_health_score,
    monthly_summary,
    top_merchants,
)
from finance.anomaly_detection import detect_anomalies
from finance.categorization import CATEGORIES
from finance.forecasting import forecast_next_month_expenses
from finance.utils import currency


@dataclass(frozen=True)
class InsightBundle:
    summary: str
    recommendations: list[str]
    recurring: pd.DataFrame


TOP_QUESTIONS = [
    "What should I do to save more?",
    "Which category did I spend the most on excluding Other?",
    "What is my most expensive transaction?",
    "Show my top 5 expenses",
    "Which merchants did I spend the most at?",
    "How much did I spend on food?",
    "Compare food spending across months",
    "What are my recurring payments?",
]


def generate_financial_insights(df: pd.DataFrame, use_openai: bool = True) -> InsightBundle:
    local_summary = generate_local_summary(df)
    recommendations = generate_recommendations(df)
    recurring = detect_recurring_expenses(df)

    if use_openai and os.getenv("OPENAI_API_KEY"):
        ai_summary = _try_openai_summary(df, local_summary, recommendations)
        if ai_summary:
            return InsightBundle(summary=ai_summary, recommendations=recommendations, recurring=recurring)

    return InsightBundle(summary=local_summary, recommendations=recommendations, recurring=recurring)


def generate_local_summary(df: pd.DataFrame) -> str:
    kpis = build_kpis(df)
    categories = category_breakdown(df)
    forecast = forecast_next_month_expenses(df)
    score, details = financial_health_score(df)
    top_category = categories.iloc[0]["category"] if not categories.empty else "no category"
    top_amount = float(categories.iloc[0]["total"]) if not categories.empty else 0.0

    return (
        f"You recorded {int(kpis['transactions'])} transactions with total income of {currency(kpis['income'])} "
        f"and expenses of {currency(kpis['expenses'])}. Your estimated savings rate is "
        f"{kpis['savings_rate']:.1f}%. The largest spending category is {top_category} at "
        f"{currency(top_amount)}. The next month expense forecast is {currency(float(forecast['forecast']))}. "
        f"Your financial health score is {score}/100, with an average savings rate of "
        f"{details['avg_savings_rate']:.1f}%."
    )


def generate_recommendations(df: pd.DataFrame) -> list[str]:
    recommendations: list[str] = []
    kpis = build_kpis(df)
    categories = category_breakdown(df)
    anomalies = detect_anomalies(df)
    recurring = detect_recurring_expenses(df)

    if kpis["income"] and kpis["savings_rate"] < 20:
        recommendations.append("Aim to move savings above 20% of income by setting a category budget.")
    if not categories.empty:
        top = categories.iloc[0]
        recommendations.append(
            f"Review {top['category']} spending first; it is the largest expense bucket at {currency(float(top['total']))}."
        )
    if not anomalies.empty:
        recommendations.append(f"Check {len(anomalies)} unusually large transaction(s) before finalizing the budget.")
    if not recurring.empty:
        recommendations.append("Audit recurring payments and cancel subscriptions that are no longer useful.")
    if not recommendations:
        recommendations.append("Spending looks stable; continue tracking monthly savings and recurring payments.")
    return recommendations


def answer_question(df: pd.DataFrame, question: str) -> str:
    q = question.lower().strip()
    if not q:
        return "Ask a question about your spending, categories, anomalies, income, or savings."

    categories = category_breakdown(df)
    summary = monthly_summary(df)

    if _asks_for_comparison(q):
        return _comparison_answer(df, q)

    if _asks_for_top_merchants(q):
        return _top_merchants_answer(df, q)

    if _asks_for_transaction(q):
        return _transaction_answer(df, q)

    if _asks_for_rows(q):
        return _rows_answer(df, q)

    if _asks_for_ranked_category(q):
        if categories.empty:
            return "There are no expenses to compare yet."
        return _ranked_category_answer(categories, q)

    if "food" in q:
        return _category_answer(categories, "Food")

    if _asks_for_total(q):
        return _total_answer(df, q)

    if "last month" in q:
        if summary.empty:
            return "No monthly data is available yet."
        last = summary.iloc[-1]
        return (
            f"In {last['month']}, income was {currency(float(last['income']))}, expenses were "
            f"{currency(float(last['expenses']))}, and savings were {currency(float(last['net_savings']))}."
        )

    if "unusual" in q or "anomal" in q:
        anomalies = detect_anomalies(df)
        if anomalies.empty:
            return "No unusual expenses were detected in the current data."
        row = anomalies.iloc[0]
        reason = row.get("anomaly_reason", "it differs from your usual spending pattern")
        return (
            f"The largest unusual expense is {currency(float(row['expense']))} at "
            f"{row['merchant']} on {pd.to_datetime(row['date']).date()}. Reason: {reason}."
        )

    if "subscription" in q or "recurring" in q:
        recurring = detect_recurring_expenses(df)
        if recurring.empty:
            return "No recurring payments were detected."
        row = recurring.iloc[0]
        return f"The most frequent recurring payment appears to be {row['merchant']} ({int(row['occurrences'])} times)."

    if _asks_for_savings_advice(q):
        return _savings_advice_answer(df)

    if "save" in q or "savings" in q:
        kpis = build_kpis(df)
        return f"Your estimated savings are {currency(kpis['savings'])}, a savings rate of {kpis['savings_rate']:.1f}%."

    if "forecast" in q or "next month" in q:
        forecast = forecast_next_month_expenses(df)
        return (
            f"The next month expense forecast is {currency(float(forecast['forecast']))} "
            f"for {forecast['next_month']} using {str(forecast['method']).replace('_', ' ')}."
        )

    if "health" in q or "score" in q:
        score, details = financial_health_score(df)
        return (
            f"Your financial health score is {score}/100. Average savings rate is "
            f"{details['avg_savings_rate']:.1f}% and expense stability is {details['expense_stability']:.1f}%."
        )

    return generate_local_summary(df)


def _asks_for_transaction(q: str) -> bool:
    transaction_words = ("transaction", "payment", "purchase", "spend", "expense")
    amount_words = ("most expensive", "largest", "highest", "biggest", "costliest", "maximum")
    return any(word in q for word in transaction_words) and any(word in q for word in amount_words)


def _asks_for_rows(q: str) -> bool:
    row_words = ("show", "list", "find", "retrieve", "get", "display")
    target_words = ("transactions", "transaction", "rows", "payments", "purchases", "expenses")
    return any(word in q for word in row_words) and any(word in q for word in target_words)


def _asks_for_ranked_category(q: str) -> bool:
    category_words = ("category", "categories", "bucket")
    rank_words = ("most", "largest", "highest", "biggest", "top", "after", "second")
    return any(word in q for word in category_words) and any(word in q for word in rank_words)


def _asks_for_comparison(q: str) -> bool:
    return any(word in q for word in ("compare", "versus", " vs ", "difference", "trend", "across months"))


def _comparison_answer(df: pd.DataFrame, q: str) -> str:
    data = add_expense_columns(df)
    data = data[data["expense"] > 0].copy()
    if data.empty:
        return "There are no expenses to compare yet."

    category = _mentioned_category(q)
    if category:
        data = data[data["category"].astype(str).str.lower() == category.lower()]
        label = f"{category} spending"
    else:
        label = "expense spending"

    months = _mentioned_months(q, data)
    if months:
        data = data[data["month"].isin(months)]
    if data.empty:
        return f"I could not find matching {label.lower()} for that comparison."

    grouped = data.groupby("month", as_index=False).agg(total=("expense", "sum")).sort_values("month")
    if len(grouped) == 1:
        row = grouped.iloc[0]
        return f"{label} in {row['month']} was {currency(float(row['total']))}."

    first = grouped.iloc[0]
    last = grouped.iloc[-1]
    change = float(last["total"] - first["total"])
    direction = "increased" if change > 0 else "decreased" if change < 0 else "stayed the same"
    table = _format_simple_table(grouped, {"total": "amount"})
    return (
        f"{label} {direction} by {currency(abs(change))} from {first['month']} to {last['month']}.\n\n"
        f"{table}"
    )


def _asks_for_top_merchants(q: str) -> bool:
    merchant_words = ("merchant", "merchants", "vendor", "vendors", "payee", "payees")
    rank_words = ("top", "most", "largest", "highest", "biggest")
    return any(word in q for word in merchant_words) and any(word in q for word in rank_words)


def _top_merchants_answer(df: pd.DataFrame, q: str) -> str:
    merchants = top_merchants(df, limit=_requested_limit(q))
    if merchants.empty:
        return "There are no expense merchants to compare yet."
    table = _format_simple_table(merchants[["merchant", "total", "transactions"]], {"total": "amount"})
    return f"These are your top spending merchants:\n\n{table}"


def _asks_for_total(q: str) -> bool:
    total_words = ("how much", "total", "sum", "spent on", "spend on", "paid for")
    return any(phrase in q for phrase in total_words)


def _total_answer(df: pd.DataFrame, q: str) -> str:
    if "income" in q or "earned" in q:
        kpis = build_kpis(df)
        return f"Your total income is {currency(kpis['income'])}."

    rows = _filter_rows(df, q, expenses_only=True)
    if rows.empty:
        return "I could not find matching expenses for that question."

    total = float(rows["expense"].sum())
    parts = []
    category = _mentioned_category(q)
    merchant = _mentioned_merchant_text(q, rows)
    months = _mentioned_months(q, rows)
    if category:
        parts.append(category)
    if merchant:
        parts.append(merchant.title())
    if months:
        parts.append(", ".join(months))
    label = " for " + " in ".join(parts) if parts else ""
    return f"You spent {currency(total)}{label} across {len(rows)} transaction(s)."


def _asks_for_savings_advice(q: str) -> bool:
    advice_words = ("what should", "how can", "how do", "advice", "recommend", "suggest", "improve", "reduce")
    savings_words = ("save", "savings", "spend less", "cut spending")
    return any(phrase in q for phrase in advice_words) and any(phrase in q for phrase in savings_words)


def _savings_advice_answer(df: pd.DataFrame) -> str:
    kpis = build_kpis(df)
    categories = category_breakdown(df)
    recurring = detect_recurring_expenses(df)
    anomalies = detect_anomalies(df)

    if categories.empty:
        return "I need expense transactions before I can suggest where to save more."

    top = categories.iloc[0]
    target_cut = float(top["total"]) * 0.1
    advice = [
        f"Your savings rate is {kpis['savings_rate']:.1f}%, so the biggest lever is reducing your largest expense bucket.",
        f"Start with {top['category']}, where you spent {currency(float(top['total']))}. A 10% reduction there would save about {currency(target_cut)}.",
    ]
    if len(categories) > 1:
        second = categories.iloc[1]
        advice.append(f"Next, review {second['category']} spending at {currency(float(second['total']))}.")
    if not recurring.empty:
        row = recurring.iloc[0]
        advice.append(f"Audit recurring payments, especially {row['merchant']} ({int(row['occurrences'])} times).")
    if not anomalies.empty:
        advice.append(f"Check {len(anomalies)} unusually large expense(s) before setting next month's budget.")
    return " ".join(advice)


def _ranked_category_answer(categories: pd.DataFrame, q: str) -> str:
    ranked = categories.copy()
    if _should_exclude_other(q):
        ranked = ranked[ranked["category"].str.lower() != "other"]
        if ranked.empty:
            return "After excluding Other, there are no categorized expenses to compare."

    rank = _requested_rank(q)
    if rank >= len(ranked):
        return f"I only found {len(ranked)} spending categor{'y' if len(ranked) == 1 else 'ies'} for that request."

    row = ranked.iloc[rank]
    prefix = "After excluding Other, " if _should_exclude_other(q) else ""
    if rank == 0:
        return f"{prefix}you spent the most on {row['category']}: {currency(float(row['total']))}."
    return f"{prefix}your #{rank + 1} spending category is {row['category']}: {currency(float(row['total']))}."


def _should_exclude_other(q: str) -> bool:
    return any(phrase in q for phrase in ("after other", "after others", "excluding other", "exclude other", "besides other"))


def _requested_rank(q: str) -> int:
    if "second" in q or "2nd" in q:
        return 1
    if "third" in q or "3rd" in q:
        return 2
    match = re.search(r"\btop\s+(\d+)\b", q)
    if match:
        return max(0, int(match.group(1)) - 1)
    return 0


def _transaction_answer(df: pd.DataFrame, q: str) -> str:
    rows = _filter_rows(df, q, expenses_only=True)
    if rows.empty:
        return "I could not find matching expenses for that question."

    row = rows.sort_values("expense", ascending=False).iloc[0]
    return (
        f"Your most expensive matching transaction was {currency(float(row['expense']))} "
        f"for {row['description']} on {pd.to_datetime(row['date']).date()} "
        f"in {row.get('category', 'Uncategorized')}."
    )


def _rows_answer(df: pd.DataFrame, q: str) -> str:
    rows = _filter_rows(df, q, expenses_only=("expense" in q or "spend" in q or "purchase" in q))
    if rows.empty:
        return "I could not find matching transactions."
    limit = _requested_limit(q)
    return _format_rows(rows.head(limit), limit)


def _filter_rows(df: pd.DataFrame, q: str, expenses_only: bool = False) -> pd.DataFrame:
    data = add_expense_columns(df)
    if expenses_only:
        data = data[data["expense"] > 0]

    category = _mentioned_category(q)
    if category:
        if "category" not in data.columns:
            return data.iloc[0:0]
        data = data[data["category"].astype(str).str.lower() == category.lower()]

    merchant = _mentioned_merchant_text(q, data)
    if merchant:
        text = merchant.lower()
        data = data[
            data["description"].astype(str).str.lower().str.contains(text, na=False)
            | data["merchant"].astype(str).str.lower().str.contains(text, na=False)
        ]

    months = _mentioned_months(q, data)
    if months:
        data = data[data["month"].isin(months)]

    amount_filter = _amount_filter(q)
    if amount_filter:
        operator, value = amount_filter
        amount_series = data["expense"] if expenses_only else data["amount"].abs()
        if operator == ">":
            data = data[amount_series > value]
        elif operator == ">=":
            data = data[amount_series >= value]
        elif operator == "<":
            data = data[amount_series < value]
        elif operator == "<=":
            data = data[amount_series <= value]

    sort_column = "expense" if "expense" in data.columns and expenses_only else "date"
    ascending = False if sort_column == "expense" else True
    return data.sort_values(sort_column, ascending=ascending)


def _mentioned_category(q: str) -> str | None:
    for category in CATEGORIES:
        if category.lower() in q:
            return category
    return None


def _quoted_text(q: str) -> str | None:
    match = re.search(r"['\"]([^'\"]+)['\"]", q)
    return match.group(1).strip() if match else None


def _mentioned_merchant_text(q: str, df: pd.DataFrame) -> str | None:
    quoted = _quoted_text(q)
    if quoted:
        return quoted
    if "merchant" not in df.columns:
        return None

    stopwords = {
        "transaction",
        "transactions",
        "expense",
        "expenses",
        "payment",
        "payments",
        "purchase",
        "purchases",
        "show",
        "list",
        "find",
        "get",
        "top",
        "above",
        "below",
        "over",
        "under",
    }
    category_words = {category.lower() for category in CATEGORIES}
    merchants = [str(value).strip() for value in df["merchant"].dropna().unique() if str(value).strip()]
    for merchant in sorted(merchants, key=len, reverse=True):
        normalized = re.sub(r"[^a-z0-9]+", " ", merchant.lower()).strip()
        if len(normalized) >= 3 and normalized in q:
            return normalized
        for token in normalized.split():
            if len(token) >= 4 and token not in stopwords and token not in category_words and token in q:
                return token
    return None


def _mentioned_months(q: str, df: pd.DataFrame) -> list[str]:
    if "month" not in df.columns:
        return []
    months = sorted(df["month"].dropna().astype(str).unique())
    matched = [month for month in months if month.lower() in q]
    if matched:
        return matched

    if "date" not in df.columns:
        return []
    dates = pd.to_datetime(df["date"], errors="coerce")
    month_names = pd.DataFrame({"month": df["month"].astype(str), "date": dates}).dropna()
    matched_months: list[str] = []
    for _, row in month_names.drop_duplicates("month").iterrows():
        full_name = row["date"].strftime("%B").lower()
        short_name = row["date"].strftime("%b").lower()
        if re.search(rf"\b{re.escape(full_name)}\b", q) or re.search(rf"\b{re.escape(short_name)}\b", q):
            matched_months.append(str(row["month"]))
    return sorted(set(matched_months))


def _amount_filter(q: str) -> tuple[str, float] | None:
    patterns = [
        (r"\b(?:above|over|more than|greater than)\s*(?:rs\.?|₹|inr)?\s*([0-9][0-9,]*(?:\.\d+)?)", ">"),
        (r"\b(?:at least|minimum|min)\s*(?:rs\.?|₹|inr)?\s*([0-9][0-9,]*(?:\.\d+)?)", ">="),
        (r"\b(?:below|under|less than|lower than)\s*(?:rs\.?|₹|inr)?\s*([0-9][0-9,]*(?:\.\d+)?)", "<"),
        (r"\b(?:at most|maximum|max)\s*(?:rs\.?|₹|inr)?\s*([0-9][0-9,]*(?:\.\d+)?)", "<="),
    ]
    for pattern, operator in patterns:
        match = re.search(pattern, q)
        if match:
            return operator, float(match.group(1).replace(",", ""))
    return None


def _format_simple_table(df: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    display = df.copy()
    for column, format_name in formats.items():
        if column in display.columns and format_name == "amount":
            display[column] = display[column].map(lambda value: currency(float(value)))
    columns = list(display.columns)
    table_rows = ["| " + " | ".join(columns) + " |"]
    table_rows.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in display.iterrows():
        table_rows.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(table_rows)


def _requested_limit(q: str) -> int:
    match = re.search(r"\b(?:top|first|last|show|list|get)\s+(\d+)\b", q)
    if match:
        return min(max(1, int(match.group(1))), 20)
    return 10


def _format_rows(rows: pd.DataFrame, limit: int) -> str:
    display = rows.copy()
    display["date"] = pd.to_datetime(display["date"]).dt.strftime("%Y-%m-%d")
    display["amount"] = display["amount"].map(lambda value: currency(float(value)))
    columns = ["date", "description", "category", "amount"]
    available_columns = [column for column in columns if column in display.columns]
    table_rows = ["| " + " | ".join(available_columns) + " |"]
    table_rows.append("| " + " | ".join("---" for _ in available_columns) + " |")
    for _, row in display[available_columns].iterrows():
        table_rows.append("| " + " | ".join(str(row[column]) for column in available_columns) + " |")
    table = "\n".join(table_rows)
    return f"Here are {min(len(display), limit)} matching transaction(s):\n\n{table}"


def _category_answer(categories: pd.DataFrame, category: str) -> str:
    row = categories[categories["category"] == category]
    if row.empty:
        return f"No {category.lower()} expenses were found."
    value = float(row.iloc[0]["total"])
    return f"You spent {currency(value)} on {category.lower()}."


def _try_openai_summary(df: pd.DataFrame, local_summary: str, recommendations: list[str]) -> str | None:
    try:
        from openai import OpenAI

        client = OpenAI()
        categories = category_breakdown(df).head(8).to_dict(orient="records")
        monthly = monthly_summary(df).tail(6).to_dict(orient="records")
        prompt = (
            "Write a concise personal finance summary in plain English. "
            "Include spending patterns, budget recommendations, and savings suggestions.\n"
            f"Baseline summary: {local_summary}\n"
            f"Recommendations: {recommendations}\n"
            f"Category data: {categories}\n"
            f"Monthly data: {monthly}"
        )
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a practical personal finance assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=350,
        )
        return response.choices[0].message.content
    except Exception:
        return None
