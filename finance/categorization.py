from __future__ import annotations

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


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

MIN_LEARNED_EXAMPLES = 6
MIN_LEARNED_CATEGORIES = 2
MIN_LEARNED_CONFIDENCE = 0.55


def categorize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    categorized = df.copy()
    existing_category = (
        categorized["category"].where(categorized["category"].isin(CATEGORIES))
        if "category" in categorized.columns
        else pd.Series(index=categorized.index, dtype=object)
    )
    categorized["rule_category"] = categorized.apply(
        lambda row: categorize_transaction(row.get("description", ""), row.get("amount", 0)), axis=1
    )
    categorized["category"] = categorized["rule_category"]
    categorized["category_source"] = "rules"
    categorized["category_confidence"] = 1.0
    has_existing_category = existing_category.notna()
    categorized.loc[has_existing_category, "category"] = existing_category.loc[has_existing_category]
    categorized.loc[has_existing_category, "category_source"] = "existing"

    learned = _learned_category_predictions(categorized)
    if learned is not None:
        categories, confidence = learned
        use_learned = (
            (categorized["rule_category"] == "Other")
            & (categorized["category_source"] != "existing")
            & (confidence >= MIN_LEARNED_CONFIDENCE)
        )
        categorized.loc[use_learned, "category"] = categories.loc[use_learned]
        categorized.loc[use_learned, "category_source"] = "learned"
        categorized.loc[use_learned, "category_confidence"] = confidence.loc[use_learned].round(2)

    categorized = categorized.drop(columns=["rule_category"])
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


def _learned_category_predictions(df: pd.DataFrame) -> tuple[pd.Series, pd.Series] | None:
    training_labels = df["category"].where(df["category"].isin(CATEGORIES), df["rule_category"])
    training = df[
        (training_labels != "Other")
        & df["description"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    training["training_category"] = training_labels.loc[training.index]
    if len(training) < MIN_LEARNED_EXAMPLES or training["training_category"].nunique() < MIN_LEARNED_CATEGORIES:
        return None

    pipeline = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    x_train = pipeline.fit_transform(training["description"].astype(str))
    model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    model.fit(x_train, training["training_category"])

    x_all = pipeline.transform(df["description"].fillna("").astype(str))
    predictions = pd.Series(model.predict(x_all), index=df.index)
    probabilities = model.predict_proba(x_all).max(axis=1)
    confidence = pd.Series(probabilities, index=df.index)
    return predictions, confidence
