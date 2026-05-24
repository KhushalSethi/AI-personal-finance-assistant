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
