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


def test_learned_categorization_uses_existing_labeled_data():
    df = pd.DataFrame(
        {
            "description": [
                "Bluebowl lunch",
                "Bluebowl dinner",
                "Bluebowl meal",
                "Citycab ride",
                "Citycab airport",
                "Citycab station",
                "Bluebowl brunch",
            ],
            "amount": [-400, -500, -450, -700, -900, -650, -550],
            "category": ["Food", "Food", "Food", "Travel", "Travel", "Travel", None],
        }
    )

    result = categorize_transactions(df)

    assert result.loc[6, "category"] == "Food"
    assert result.loc[6, "category_source"] == "learned"
    assert result.loc[6, "category_confidence"] >= 0.55
