from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from finance.analytics import monthly_summary


def forecast_next_month_expenses(df: pd.DataFrame) -> dict[str, float | str]:
    summary = monthly_summary(df)
    if summary.empty:
        return {"next_month": "N/A", "forecast": 0.0, "method": "no_data"}

    periods = pd.PeriodIndex(summary["month"], freq="M")
    next_month = str(periods.max() + 1)

    if len(summary) < 3:
        return {
            "next_month": next_month,
            "forecast": round(float(summary["expenses"].mean()), 2),
            "method": "average",
        }

    x = list(range(len(summary)))
    model = LinearRegression()
    model.fit(pd.DataFrame({"month_index": x}), summary["expenses"])
    prediction = float(model.predict(pd.DataFrame({"month_index": [len(summary)]}))[0])
    rolling_avg = float(summary["expenses"].tail(3).mean())
    forecast = max(0.0, (prediction + rolling_avg) / 2)
    return {"next_month": next_month, "forecast": round(forecast, 2), "method": "linear_regression"}


def evaluate_forecast_models(df: pd.DataFrame, min_train_months: int = 3) -> pd.DataFrame:
    summary = monthly_summary(df)
    if len(summary) <= min_train_months:
        return pd.DataFrame(columns=["model", "mae", "mape", "evaluated_months"])

    expenses = summary["expenses"].astype(float).reset_index(drop=True)
    results: dict[str, list[tuple[float, float]]] = {
        "average": [],
        "moving_average": [],
        "linear_regression": [],
        "random_forest": [],
    }

    for index in range(min_train_months, len(expenses)):
        train = expenses.iloc[:index]
        actual = float(expenses.iloc[index])
        predictions = {
            "average": float(train.mean()),
            "moving_average": float(train.tail(3).mean()),
            "linear_regression": _linear_prediction(train),
            "random_forest": _random_forest_prediction(train),
        }
        for model_name, predicted in predictions.items():
            absolute_error = abs(actual - predicted)
            percent_error = (absolute_error / actual * 100) if actual else 0.0
            results[model_name].append((absolute_error, percent_error))

    rows = []
    for model_name, errors in results.items():
        if not errors:
            continue
        rows.append(
            {
                "model": model_name,
                "mae": round(float(pd.Series([error[0] for error in errors]).mean()), 2),
                "mape": round(float(pd.Series([error[1] for error in errors]).mean()), 2),
                "evaluated_months": len(errors),
            }
        )
    return pd.DataFrame(rows).sort_values(["mae", "mape"]).reset_index(drop=True)


def _linear_prediction(train: pd.Series) -> float:
    x = pd.DataFrame({"month_index": range(len(train))})
    model = LinearRegression()
    model.fit(x, train)
    return max(0.0, float(model.predict(pd.DataFrame({"month_index": [len(train)]}))[0]))


def _random_forest_prediction(train: pd.Series) -> float:
    x = pd.DataFrame({"month_index": range(len(train))})
    model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=1)
    model.fit(x, train)
    return max(0.0, float(model.predict(pd.DataFrame({"month_index": [len(train)]}))[0]))
