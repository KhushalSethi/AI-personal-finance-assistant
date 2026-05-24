from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import IO

import pandas as pd

logger = logging.getLogger(__name__)


STANDARD_COLUMNS = ["date", "description", "amount", "merchant", "transaction_type"]

DATE_COLUMNS = {"date", "transaction date", "posted date", "posting date", "value date"}
DESCRIPTION_COLUMNS = {"description", "details", "narration", "memo", "merchant", "payee"}
AMOUNT_COLUMNS = {"amount", "transaction amount", "amt", "value"}
DEBIT_COLUMNS = {"debit", "withdrawal", "withdrawals", "spent", "expense"}
CREDIT_COLUMNS = {"credit", "deposit", "deposits", "income", "received"}


class CSVProcessingError(ValueError):
    """Raised when a CSV cannot be normalized into transactions."""


@dataclass(frozen=True)
class ProcessingResult:
    transactions: pd.DataFrame
    warnings: list[str]


def read_csv_safely(file: str | IO[bytes] | IO[str]) -> ProcessingResult:
    warnings: list[str] = []
    try:
        raw = pd.read_csv(file)
    except UnicodeDecodeError:
        try:
            raw = pd.read_csv(file, encoding="latin-1")
            warnings.append("CSV was read with latin-1 encoding.")
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to read CSV with fallback encoding")
            raise CSVProcessingError("Could not read the CSV file.") from exc
    except pd.errors.EmptyDataError as exc:
        raise CSVProcessingError("The CSV file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise CSVProcessingError("The CSV file appears malformed.") from exc
    except Exception as exc:
        logger.exception("Unexpected CSV read failure")
        raise CSVProcessingError("Could not read the CSV file.") from exc

    if raw.empty:
        raise CSVProcessingError("The CSV file does not contain any transactions.")

    return ProcessingResult(*preprocess_transactions(raw, warnings=warnings))


def preprocess_transactions(
    df: pd.DataFrame, warnings: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    warnings = list(warnings or [])
    data = df.copy()
    data.columns = [_normalize_column(c) for c in data.columns]

    date_col = _first_matching_column(data, DATE_COLUMNS)
    description_col = _first_matching_column(data, DESCRIPTION_COLUMNS)
    amount_col = _first_matching_column(data, AMOUNT_COLUMNS)
    debit_col = _first_matching_column(data, DEBIT_COLUMNS)
    credit_col = _first_matching_column(data, CREDIT_COLUMNS)

    if not date_col:
        raise CSVProcessingError("A date column is required.")
    if not description_col:
        warnings.append("No description column found; using blank descriptions.")
        data["description"] = ""
        description_col = "description"

    if amount_col:
        amount = _parse_money(data[amount_col])
    elif debit_col or credit_col:
        debit = _parse_money(data[debit_col]) if debit_col else pd.Series(0, index=data.index)
        credit = _parse_money(data[credit_col]) if credit_col else pd.Series(0, index=data.index)
        amount = credit.fillna(0) - debit.fillna(0)
    else:
        raise CSVProcessingError("An amount column or debit/credit columns are required.")

    dates = pd.to_datetime(data[date_col], errors="coerce", dayfirst=False)
    missing_dates = dates.isna()
    if missing_dates.any():
        retry_dates = pd.to_datetime(data.loc[missing_dates, date_col], errors="coerce", dayfirst=True)
        dates.loc[missing_dates] = retry_dates

    cleaned = pd.DataFrame(
        {
            "date": dates,
            "description": data[description_col].fillna("").astype(str).str.strip(),
            "amount": amount,
        }
    )
    cleaned = cleaned.dropna(subset=["date", "amount"])
    dropped = len(data) - len(cleaned)
    if dropped:
        warnings.append(f"Dropped {dropped} rows with invalid dates or amounts.")

    cleaned["description"] = cleaned["description"].replace("", "Unknown transaction")
    cleaned["merchant"] = cleaned["description"].map(extract_merchant)
    cleaned["transaction_type"] = cleaned["amount"].map(lambda value: "income" if value > 0 else "expense")
    cleaned["month"] = cleaned["date"].dt.to_period("M").astype(str)
    cleaned["year"] = cleaned["date"].dt.year
    cleaned["day"] = cleaned["date"].dt.day
    cleaned = cleaned.sort_values("date").reset_index(drop=True)
    cleaned = cleaned[STANDARD_COLUMNS + ["month", "year", "day"]]

    if cleaned.empty:
        raise CSVProcessingError("No valid transaction rows were found.")

    return cleaned, warnings


def extract_merchant(description: str) -> str:
    text = " ".join(str(description).replace("*", " ").replace("-", " ").split())
    if not text:
        return "Unknown"
    prefixes = {"pos", "upi", "neft", "imps", "ach", "debit", "credit", "card", "payment"}
    words = [word for word in text.split() if word.lower().strip(":") not in prefixes]
    merchant = " ".join(words[:4]) if words else text
    return merchant.title()


def _normalize_column(column: object) -> str:
    return str(column).strip().lower().replace("_", " ")


def _first_matching_column(df: pd.DataFrame, choices: set[str]) -> str | None:
    for column in df.columns:
        if column in choices:
            return column
    return None


def _parse_money(series: pd.Series) -> pd.Series:
    as_text = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.strip()
    )
    negative_parentheses = as_text.str.match(r"^\(.*\)$", na=False)
    as_text = as_text.str.replace(r"[()]", "", regex=True)
    values = pd.to_numeric(as_text, errors="coerce")
    values.loc[negative_parentheses] = -values.loc[negative_parentheses].abs()
    return values
