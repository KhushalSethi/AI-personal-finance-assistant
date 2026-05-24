from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "finance.db"


def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(exist_ok=True)


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    ensure_data_dir()
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            description TEXT NOT NULL,
            amount REAL NOT NULL,
            merchant TEXT,
            transaction_type TEXT,
            month TEXT,
            year INTEGER,
            day INTEGER,
            category TEXT
        )
        """
    )
    return conn


def save_transactions(df: pd.DataFrame, db_path: Path = DB_PATH) -> int:
    if df.empty:
        return 0
    columns = [
        "date",
        "description",
        "amount",
        "merchant",
        "transaction_type",
        "month",
        "year",
        "day",
        "category",
    ]
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
    with get_connection(db_path) as conn:
        data[columns].to_sql("transactions", conn, if_exists="append", index=False)
    return len(data)


def load_transactions(db_path: Path = DB_PATH) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM transactions ORDER BY date", conn)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def currency(value: float) -> str:
    return f"₹{value:,.2f}"
