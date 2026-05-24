from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from finance.analytics import add_expense_columns


def cluster_monthly_spending_patterns(df: pd.DataFrame, max_clusters: int = 3) -> pd.DataFrame:
    data = add_expense_columns(df)
    expenses = data[data["expense"] > 0].copy()
    if expenses.empty or expenses["month"].nunique() < 2:
        return pd.DataFrame(
            columns=["month", "cluster", "profile", "total_expenses", "dominant_category", "dominant_category_share"]
        )

    if "category" not in expenses.columns:
        expenses["category"] = "Other"

    monthly_categories = expenses.pivot_table(
        index="month",
        columns="category",
        values="expense",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()
    totals = monthly_categories.sum(axis=1)
    shares = monthly_categories.div(totals.replace(0, 1), axis=0)

    cluster_count = min(max_clusters, len(monthly_categories))
    features = StandardScaler().fit_transform(shares)
    labels = KMeans(n_clusters=cluster_count, random_state=42, n_init=10).fit_predict(features)

    clustered = pd.DataFrame(
        {
            "month": monthly_categories.index,
            "cluster": labels,
            "total_expenses": totals.values,
            "dominant_category": monthly_categories.idxmax(axis=1).values,
            "dominant_category_share": shares.max(axis=1).round(3).values,
        }
    )
    clustered["profile"] = clustered.apply(_cluster_profile, axis=1)
    return clustered.sort_values(["cluster", "month"]).reset_index(drop=True)


def summarize_spending_clusters(clusters: pd.DataFrame) -> str:
    if clusters.empty:
        return "At least two months of expense data are needed for spending pattern clustering."

    profiles = []
    for profile, group in clusters.groupby("profile"):
        months = ", ".join(group["month"].astype(str))
        avg_expense = float(group["total_expenses"].mean())
        profiles.append(f"{profile}: {months} with average expenses of ₹{avg_expense:,.2f}")
    return " | ".join(profiles)


def _cluster_profile(row: pd.Series) -> str:
    share = float(row["dominant_category_share"])
    category = str(row["dominant_category"])
    if share >= 0.5:
        return f"{category}-heavy month"
    return "mixed-spending month"
