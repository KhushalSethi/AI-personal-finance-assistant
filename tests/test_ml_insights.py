import pandas as pd

from finance.forecasting import evaluate_forecast_models
from finance.ml_insights import cluster_monthly_spending_patterns, summarize_spending_clusters


def monthly_df():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-02-01",
                    "2026-02-02",
                    "2026-03-01",
                    "2026-03-02",
                    "2026-04-01",
                    "2026-04-02",
                    "2026-05-01",
                    "2026-05-02",
                ]
            ),
            "amount": [-5000, -500, -5200, -600, -1000, -4200, -1200, -4500, -2000, -2500],
            "category": [
                "Bills",
                "Food",
                "Bills",
                "Food",
                "Bills",
                "Travel",
                "Bills",
                "Travel",
                "Food",
                "Shopping",
            ],
            "merchant": [
                "Rent",
                "Cafe",
                "Rent",
                "Cafe",
                "Rent",
                "Flight",
                "Rent",
                "Flight",
                "Restaurant",
                "Store",
            ],
            "month": ["2026-01", "2026-01", "2026-02", "2026-02", "2026-03", "2026-03", "2026-04", "2026-04", "2026-05", "2026-05"],
            "day": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        }
    )


def test_cluster_monthly_spending_patterns_returns_profiles():
    clusters = cluster_monthly_spending_patterns(monthly_df())

    assert not clusters.empty
    assert {"month", "cluster", "profile", "dominant_category"}.issubset(clusters.columns)
    assert clusters["month"].nunique() == 5
    assert "month" in summarize_spending_clusters(clusters)


def test_evaluate_forecast_models_ranks_models_by_error():
    scores = evaluate_forecast_models(monthly_df())

    assert not scores.empty
    assert scores.iloc[0]["mae"] <= scores.iloc[-1]["mae"]
    assert set(scores["model"]) >= {"average", "moving_average", "linear_regression"}
