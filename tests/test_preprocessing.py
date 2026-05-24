import pandas as pd
import pytest

from finance.preprocessing import CSVProcessingError, preprocess_transactions


def test_preprocess_amount_column_and_dates():
    raw = pd.DataFrame(
        {
            "Transaction Date": ["2026-01-02", "03/01/2026", "bad-date"],
            "Description": ["Coffee Cafe", "Salary Payroll", "Broken"],
            "Amount": ["-120.50", "5000", "10"],
        }
    )

    cleaned, warnings = preprocess_transactions(raw)

    assert len(cleaned) == 2
    assert cleaned.loc[0, "merchant"] == "Coffee Cafe"
    assert cleaned.loc[1, "transaction_type"] == "income"
    assert warnings


def test_preprocess_debit_credit_columns():
    raw = pd.DataFrame(
        {
            "Date": ["01-02-2026", "02-02-2026"],
            "Narration": ["Rent Bill", "Salary"],
            "Debit": ["1000", ""],
            "Credit": ["", "5000"],
        }
    )

    cleaned, _ = preprocess_transactions(raw)

    assert cleaned["amount"].tolist() == [-1000.0, 5000.0]


def test_preprocess_requires_amount_information():
    raw = pd.DataFrame({"Date": ["2026-01-01"], "Description": ["Coffee"]})

    with pytest.raises(CSVProcessingError):
        preprocess_transactions(raw)
