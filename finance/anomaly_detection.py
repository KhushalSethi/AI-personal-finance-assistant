from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest

from finance.analytics import add_expense_columns


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    data = add_expense_columns(df)
    expenses = data[data["expense"] > 0].copy()
    if len(expenses) < 6:
        expenses["anomaly_score"] = 0.0
        median = expenses["expense"].median()
        mad = (expenses["expense"] - median).abs().median()
        threshold = median + max(3 * mad, median)
        expenses["is_anomaly"] = expenses["expense"] >= threshold
        return expenses[expenses["is_anomaly"]].sort_values("expense", ascending=False)

    features = expenses[["expense", "day"]].copy()
    category_means = expenses.groupby("category")["expense"].transform("mean")
    features["category_ratio"] = expenses["expense"] / category_means.replace(0, 1)

    model = IsolationForest(contamination=contamination, random_state=42)
    predictions = model.fit_predict(features)
    expenses["anomaly_score"] = model.decision_function(features)
    expenses["is_anomaly"] = predictions == -1
    return expenses[expenses["is_anomaly"]].sort_values("expense", ascending=False)
