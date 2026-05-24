import pandas as pd

from finance.anomaly_detection import detect_anomalies


def test_anomaly_detection_flags_large_small_dataset_expense():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]),
            "amount": [-100, -110, -95, -105, -1000],
            "category": ["Food"] * 5,
            "merchant": ["Cafe"] * 5,
            "month": ["2026-01"] * 5,
            "day": [1, 2, 3, 4, 5],
        }
    )

    anomalies = detect_anomalies(df)

    assert not anomalies.empty
    assert anomalies.iloc[0]["expense"] == 1000


def test_personalized_anomaly_reasons_include_category_and_merchant_behavior():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01 10:00",
                    "2026-01-02 11:00",
                    "2026-01-03 12:00",
                    "2026-01-04 13:00",
                    "2026-01-05 14:00",
                    "2026-01-06 00:30",
                ]
            ),
            "amount": [-500, -520, -480, -510, -495, -12000],
            "category": ["Food"] * 6,
            "merchant": ["Cafe"] * 5 + ["New Midnight Diner"],
            "month": ["2026-01"] * 6,
            "day": [1, 2, 3, 4, 5, 6],
        }
    )

    anomalies = detect_anomalies(df)

    assert not anomalies.empty
    reason = anomalies.iloc[0]["anomaly_reason"]
    assert "usual category amount" in reason
    assert "merchant has not appeared before" in reason
    assert "late-night" in reason
