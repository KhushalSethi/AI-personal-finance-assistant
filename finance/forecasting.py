from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression

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
