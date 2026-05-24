import pandas as pd

from finance.chatbot import answer_question


def sample_df():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-02-04",
                    "2026-02-05",
                ]
            ),
            "description": [
                "Salary Payroll",
                "Unknown vendor",
                "Apartment Rent",
                "Swiggy Dinner",
                "Amazon Purchase",
                "Zomato Dinner",
                "Amazon Accessories",
            ],
            "amount": [100000, -9000, -35000, -1200, -5000, -2000, -800],
            "merchant": [
                "Employer",
                "Unknown Vendor",
                "Apartment Rent",
                "Swiggy Dinner",
                "Amazon Purchase",
                "Zomato Dinner",
                "Amazon Accessories",
            ],
            "category": ["Salary", "Other", "Bills", "Food", "Shopping", "Food", "Shopping"],
            "month": ["2026-01", "2026-01", "2026-01", "2026-01", "2026-01", "2026-02", "2026-02"],
            "day": [1, 2, 3, 4, 5, 4, 5],
        }
    )


def test_most_category_after_other_excludes_other():
    answer = answer_question(sample_df(), "Which category did I spend the most after others?")

    assert "Bills" in answer
    assert "spent the most on Bills" in answer


def test_most_expensive_transaction_returns_transaction_not_category():
    answer = answer_question(sample_df(), "What is my most expensive transaction?")

    assert "Apartment Rent" in answer
    assert "Bills" in answer
    assert "most on" not in answer


def test_retrieve_specific_category_rows():
    answer = answer_question(sample_df(), "show food transactions")

    assert "Swiggy Dinner" in answer
    assert "| date | description | category | amount |" in answer


def test_savings_advice_returns_recommendations_not_only_kpis():
    answer = answer_question(sample_df(), "What should I do to save more?")

    assert "Start with Bills" in answer
    assert "savings rate" in answer
    assert "Your estimated savings are" not in answer


def test_unquoted_merchant_and_amount_filter_retrieve_rows():
    answer = answer_question(sample_df(), "show amazon transactions above 1000")

    assert "Amazon Purchase" in answer
    assert "Amazon Accessories" not in answer


def test_top_merchants_question_returns_merchant_table():
    answer = answer_question(sample_df(), "Which merchants did I spend the most at?")

    assert "top spending merchants" in answer
    assert "Apartment Rent" in answer


def test_compare_category_spending_across_months():
    answer = answer_question(sample_df(), "Compare food spending across months")

    assert "Food spending increased by" in answer
    assert "2026-01" in answer
    assert "2026-02" in answer
