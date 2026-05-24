# AI Personal Finance Assistant

A Python-only Streamlit app for uploading CSV bank statements, cleaning transactions, categorizing spending, finding anomalies, forecasting expenses, and generating plain-English financial insights.

For the full detailed project documentation, see [DOCUMENTATION.md](DOCUMENTATION.md).

## File Tree

```text
.
├── app.py
├── data/
│   └── sample_transactions.csv
├── finance/
│   ├── __init__.py
│   ├── analytics.py
│   ├── anomaly_detection.py
│   ├── categorization.py
│   ├── chatbot.py
│   ├── forecasting.py
│   ├── preprocessing.py
│   └── utils.py
├── requirements.txt
├── README.md
└── tests/
    ├── test_analytics.py
    ├── test_anomaly_detection.py
    ├── test_categorization.py
    └── test_preprocessing.py
```

## Features

- CSV bank statement upload with flexible column names.
- Robust preprocessing for missing values, malformed rows, multiple date formats, debit/credit columns, and currency-formatted amounts.
- Automatic rule-based transaction categorization with manual correction in the UI.
- Dashboard for monthly spending, category breakdown, income vs expenses, savings rate, top merchants, and recurring payments.
- Anomaly detection with scikit-learn `IsolationForest` and a small-data fallback.
- Next-month expense forecasting with linear regression and average fallback.
- Financial health score.
- Search, filter, and CSV export.
- Optional SQLite persistence.
- Optional OpenAI-generated summaries with deterministic local fallback.
- Simple finance chatbot for common spending questions.
- Light/dark mode toggle.

## Setup

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## API Key Setup

AI summaries work without an API key using local deterministic summaries. To enable OpenAI-generated summaries:

```bash
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_MODEL="gpt-4o-mini"
```

Then enable **Use OpenAI summaries** in the Streamlit sidebar.

## Run Locally

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit. You can test the app with:

```text
data/sample_transactions.csv
```

## Example CSV Format

The app accepts either an `Amount` column where expenses are negative and income is positive:

```csv
Date,Description,Amount
2026-01-01,Salary Payroll,120000
2026-01-02,Apartment Rent,-35000
2026-01-03,Swiggy Food Order,-850
```

It also accepts separate debit and credit columns:

```csv
Date,Narration,Debit,Credit
2026-01-02,Rent Bill,35000,
2026-01-01,Salary Payroll,,120000
```

## Tests

```bash
pytest
```

The tests cover CSV preprocessing, categorization, analytics calculations, and anomaly detection.

## Screenshots

Add screenshots here after running the app locally.

## Notes

- SQLite data is stored at `data/finance.db` when you click **Save current transactions to SQLite**.
- PDF/OCR support, multi-user login, and investment tracking are natural next additions, but the current implementation focuses on a reliable CSV-first workflow.
