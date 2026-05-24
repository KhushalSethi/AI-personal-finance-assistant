import pandas as pd

from finance.categorization import categorize_transaction, categorize_transactions


def test_keyword_categorization():
    assert categorize_transaction("Swiggy dinner order", -500) == "Food"
    assert categorize_transaction("Uber airport", -900) == "Travel"
    assert categorize_transaction("Salary payroll", 10000) == "Salary"


def test_categorize_dataframe_adds_category():
    df = pd.DataFrame({"description": ["Amazon store", "Doctor clinic"], "amount": [-100, -200]})

    result = categorize_transactions(df)

    assert result["category"].tolist() == ["Shopping", "Healthcare"]
