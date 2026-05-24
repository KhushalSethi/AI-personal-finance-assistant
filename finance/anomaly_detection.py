from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest

from finance.analytics import add_expense_columns


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    data = add_expense_columns(df)
    expenses = data[data["expense"] > 0].copy()
    if expenses.empty:
        return expenses

    expenses = _add_personalized_anomaly_features(expenses)
    if len(expenses) < 6:
        expenses["anomaly_score"] = expenses["personal_anomaly_score"]
        median = float(expenses["expense"].median())
        mad = float((expenses["expense"] - median).abs().median())
        threshold = median + max(3 * mad, median)
        expenses["is_anomaly"] = (expenses["expense"] >= threshold) | (expenses["personal_anomaly_score"] >= 2)
        return expenses[expenses["is_anomaly"]].sort_values("expense", ascending=False)

    features = expenses[["expense", "day", "merchant_frequency", "category_ratio", "merchant_ratio"]].copy()
    if "hour" in expenses.columns:
        features["hour"] = expenses["hour"]
        features["is_late_night"] = expenses["is_late_night"].astype(int)

    model = IsolationForest(contamination=contamination, random_state=42)
    predictions = model.fit_predict(features)
    expenses["isolation_score"] = model.decision_function(features)
    expenses["anomaly_score"] = expenses["personal_anomaly_score"] - expenses["isolation_score"]
    expenses["is_anomaly"] = (predictions == -1) | (expenses["personal_anomaly_score"] >= 2)
    return expenses[expenses["is_anomaly"]].sort_values(["personal_anomaly_score", "expense"], ascending=False)


def _add_personalized_anomaly_features(expenses: pd.DataFrame) -> pd.DataFrame:
    data = expenses.copy()
    if "category" not in data.columns:
        data["category"] = "Other"
    if "merchant" not in data.columns:
        data["merchant"] = "Unknown"
    data["category"] = data["category"].fillna("Other")
    data["merchant"] = data["merchant"].fillna("Unknown")

    category_medians = data.groupby("category")["expense"].transform("median")
    category_mad = data.groupby("category")["expense"].transform(lambda values: (values - values.median()).abs().median())
    data["category_ratio"] = data["expense"] / category_medians.replace(0, 1)
    data["category_threshold"] = category_medians + (3 * category_mad).where(category_mad > 0, category_medians)

    merchant_counts = data.groupby("merchant")["merchant"].transform("count")
    merchant_medians = data.groupby("merchant")["expense"].transform("median")
    merchant_mad = data.groupby("merchant")["expense"].transform(lambda values: (values - values.median()).abs().median())
    data["merchant_frequency"] = merchant_counts
    data["merchant_ratio"] = data["expense"] / merchant_medians.replace(0, 1)
    data["merchant_threshold"] = merchant_medians + (3 * merchant_mad).where(merchant_mad > 0, merchant_medians)

    data["is_new_merchant"] = data["merchant_frequency"] == 1
    data["is_category_deviation"] = (data["expense"] >= data["category_threshold"]) & (data["category_ratio"] >= 2)
    data["is_merchant_deviation"] = (
        (data["merchant_frequency"] >= 2)
        & (data["expense"] >= data["merchant_threshold"])
        & (data["merchant_ratio"] >= 2)
    )

    date_values = pd.to_datetime(data["date"], errors="coerce")
    has_time_information = bool(((date_values.dt.hour != 0) | (date_values.dt.minute != 0) | (date_values.dt.second != 0)).any())
    if has_time_information:
        data["hour"] = date_values.dt.hour.fillna(12)
        data["is_late_night"] = data["hour"].between(0, 5)
    else:
        data["is_late_night"] = False

    data["personal_anomaly_score"] = (
        data["is_new_merchant"].astype(int)
        + data["is_category_deviation"].astype(int)
        + data["is_merchant_deviation"].astype(int)
        + data["is_late_night"].astype(int)
    )
    data["anomaly_reason"] = data.apply(_build_anomaly_reason, axis=1)
    return data


def _build_anomaly_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if bool(row.get("is_category_deviation", False)):
        reasons.append(
            f"{row['category']} spend is {float(row['category_ratio']):.1f}x your usual category amount"
        )
    if bool(row.get("is_merchant_deviation", False)):
        reasons.append(
            f"{row['merchant']} spend is {float(row['merchant_ratio']):.1f}x your usual merchant amount"
        )
    if bool(row.get("is_new_merchant", False)):
        reasons.append("merchant has not appeared before in this dataset")
    if bool(row.get("is_late_night", False)):
        reasons.append("transaction happened during late-night hours")
    return "; ".join(reasons) if reasons else "flagged by isolation forest"
