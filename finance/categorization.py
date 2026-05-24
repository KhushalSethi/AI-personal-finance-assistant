from __future__ import annotations

import re

import pandas as pd


CATEGORIES = [
    "Food",
    "Travel",
    "Shopping",
    "Bills",
    "Entertainment",
    "Healthcare",
    "Salary",
    "Investments",
    "Other",
]

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Food": ("restaurant", "cafe", "coffee", "swiggy", "zomato", "grocery", "supermarket", "food"),
    "Travel": ("uber", "ola", "taxi", "fuel", "petrol", "flight", "rail", "metro", "hotel", "travel"),
    "Shopping": ("amazon", "flipkart", "myntra", "store", "mall", "shop", "retail", "purchase"),
    "Bills": ("electric", "utility", "phone", "internet", "rent", "insurance", "bill", "gas", "water"),
    "Entertainment": ("netflix", "spotify", "movie", "cinema", "prime", "hotstar", "game", "concert"),
    "Healthcare": ("pharmacy", "hospital", "clinic", "doctor", "medical", "health", "diagnostic"),
    "Salary": ("salary", "payroll", "wages", "stipend"),
    "Investments": ("mutual fund", "sip", "broker", "zerodha", "groww", "stock", "investment", "dividend"),
}


def categorize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    categorized = df.copy()
    categorized["category"] = categorized.apply(
        lambda row: categorize_transaction(row.get("description", ""), row.get("amount", 0)), axis=1
    )
    return categorized


def categorize_transaction(description: str, amount: float = 0) -> str:
    text = str(description).lower()
    if amount > 0 and any(keyword in text for keyword in CATEGORY_KEYWORDS["Salary"]):
        return "Salary"
    if amount > 0 and any(keyword in text for keyword in CATEGORY_KEYWORDS["Investments"]):
        return "Investments"

    for category, keywords in CATEGORY_KEYWORDS.items():
        if category in {"Salary", "Investments"} and amount < 0:
            continue
        if any(_contains_keyword(text, keyword) for keyword in keywords):
            return category

    if amount > 0:
        return "Salary"
    return "Other"


def apply_manual_categories(df: pd.DataFrame, corrections: dict[int, str]) -> pd.DataFrame:
    updated = df.copy()
    for index, category in corrections.items():
        if index in updated.index and category in CATEGORIES:
            updated.loc[index, "category"] = category
    return updated


def _contains_keyword(text: str, keyword: str) -> bool:
    if " " in keyword:
        return keyword in text
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None
