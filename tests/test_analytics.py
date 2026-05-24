import pandas as pd

from finance.analytics import category_breakdown, financial_health_score, monthly_summary


def sample_df():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-02-01", "2026-02-02"]),
            "amount": [10000, -2000, 12000, -3000],
            "category": ["Salary", "Food", "Salary", "Travel"],
            "merchant": ["Employer", "Cafe", "Employer", "Uber"],
            "month": ["2026-01", "2026-01", "2026-02", "2026-02"],
            "day": [1, 2, 1, 2],
        }
    )


def test_monthly_summary_calculates_savings():
    summary = monthly_summary(sample_df())

    assert summary.loc[0, "income"] == 10000
    assert summary.loc[0, "expenses"] == 2000
    assert summary.loc[0, "net_savings"] == 8000


def test_category_breakdown_uses_expenses_only():
    breakdown = category_breakdown(sample_df())

    assert breakdown["category"].tolist() == ["Travel", "Food"]
    assert breakdown["total"].sum() == 5000


def test_financial_health_score_range():
    score, details = financial_health_score(sample_df())

    assert 0 <= score <= 100
    assert "avg_savings_rate" in details
