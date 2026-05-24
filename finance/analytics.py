from __future__ import annotations

import pandas as pd


def add_expense_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["expense"] = data["amount"].where(data["amount"] < 0, 0).abs()
    data["income"] = data["amount"].where(data["amount"] > 0, 0)
    if "month" not in data.columns:
        data["month"] = pd.to_datetime(data["date"]).dt.to_period("M").astype(str)
    return data


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    data = add_expense_columns(df)
    summary = (
        data.groupby("month", as_index=False)
        .agg(income=("income", "sum"), expenses=("expense", "sum"), transactions=("amount", "count"))
        .sort_values("month")
    )
    summary["net_savings"] = summary["income"] - summary["expenses"]
    summary["savings_rate"] = summary.apply(
        lambda row: (row["net_savings"] / row["income"] * 100) if row["income"] else 0, axis=1
    )
    return summary


def category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    data = add_expense_columns(df)
    if "category" not in data.columns:
        data["category"] = "Other"
    return (
        data[data["expense"] > 0]
        .groupby("category", as_index=False)
        .agg(total=("expense", "sum"), transactions=("expense", "count"))
        .sort_values("total", ascending=False)
    )


def top_merchants(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    data = add_expense_columns(df)
    return (
        data[data["expense"] > 0]
        .groupby("merchant", as_index=False)
        .agg(total=("expense", "sum"), transactions=("expense", "count"))
        .sort_values("total", ascending=False)
        .head(limit)
    )


def income_vs_expenses(df: pd.DataFrame) -> pd.DataFrame:
    summary = monthly_summary(df)
    return summary[["month", "income", "expenses", "net_savings", "savings_rate"]]


def detect_recurring_expenses(df: pd.DataFrame, min_occurrences: int = 2) -> pd.DataFrame:
    data = add_expense_columns(df)
    expenses = data[data["expense"] > 0].copy()
    if expenses.empty:
        return pd.DataFrame(columns=["merchant", "category", "median_amount", "occurrences", "months"])

    expenses["rounded_amount"] = expenses["expense"].round(-1)
    grouped = (
        expenses.groupby(["merchant", "category", "rounded_amount"], as_index=False)
        .agg(occurrences=("expense", "count"), months=("month", lambda values: ", ".join(sorted(set(values)))))
    )
    recurring = grouped[grouped["occurrences"] >= min_occurrences].copy()
    recurring = recurring.rename(columns={"rounded_amount": "median_amount"})
    return recurring.sort_values(["occurrences", "median_amount"], ascending=[False, False])


def financial_health_score(df: pd.DataFrame) -> tuple[int, dict[str, float]]:
    summary = monthly_summary(df)
    if summary.empty:
        return 0, {"avg_savings_rate": 0, "expense_stability": 0, "income_months": 0}

    avg_savings_rate = float(summary["savings_rate"].mean())
    expense_mean = float(summary["expenses"].mean())
    expense_std = float(summary["expenses"].std(ddof=0)) if len(summary) > 1 else 0
    stability = max(0.0, 100.0 - (expense_std / expense_mean * 100 if expense_mean else 100.0))
    income_months = float((summary["income"] > 0).mean() * 100)

    score = 0
    score += min(40, max(0, avg_savings_rate) * 1.6)
    score += min(30, stability * 0.3)
    score += min(20, income_months * 0.2)
    score += 10 if summary["net_savings"].tail(1).iloc[0] >= 0 else 0
    details = {
        "avg_savings_rate": round(avg_savings_rate, 2),
        "expense_stability": round(stability, 2),
        "income_months": round(income_months, 2),
    }
    return int(round(max(0, min(100, score)))), details


def build_kpis(df: pd.DataFrame) -> dict[str, float]:
    data = add_expense_columns(df)
    income = float(data["income"].sum())
    expenses = float(data["expense"].sum())
    savings = income - expenses
    savings_rate = (savings / income * 100) if income else 0.0
    return {
        "income": income,
        "expenses": expenses,
        "savings": savings,
        "savings_rate": savings_rate,
        "transactions": float(len(data)),
    }
