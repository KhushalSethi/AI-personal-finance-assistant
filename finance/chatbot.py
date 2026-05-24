from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

from finance.analytics import (
    build_kpis,
    category_breakdown,
    detect_recurring_expenses,
    financial_health_score,
    monthly_summary,
    top_merchants,
)
from finance.anomaly_detection import detect_anomalies
from finance.forecasting import forecast_next_month_expenses
from finance.utils import currency


@dataclass(frozen=True)
class InsightBundle:
    summary: str
    recommendations: list[str]
    recurring: pd.DataFrame


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

    if "most" in q or "largest" in q or "highest" in q:
        if categories.empty:
            return "There are no expenses to compare yet."
        top = categories.iloc[0]
        return f"You spent the most on {top['category']}: {currency(float(top['total']))}."

    if "food" in q:
        return _category_answer(categories, "Food")

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
        return (
            f"The largest unusual expense is {currency(float(row['expense']))} at "
            f"{row['merchant']} on {pd.to_datetime(row['date']).date()}."
        )

    if "subscription" in q or "recurring" in q:
        recurring = detect_recurring_expenses(df)
        if recurring.empty:
            return "No recurring payments were detected."
        row = recurring.iloc[0]
        return f"The most frequent recurring payment appears to be {row['merchant']} ({int(row['occurrences'])} times)."

    if "save" in q or "savings" in q:
        kpis = build_kpis(df)
        return f"Your estimated savings are {currency(kpis['savings'])}, a savings rate of {kpis['savings_rate']:.1f}%."

    return generate_local_summary(df)


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
