# AI Personal Finance Assistant Documentation

This document explains the complete project structure, runtime flow, modules, setup, usage, testing, and troubleshooting for the AI Personal Finance Assistant.

## 1. Project Overview

The AI Personal Finance Assistant is a Python-only Streamlit application that helps users upload CSV bank statements and receive structured financial analytics.

The app can:

- Upload CSV bank statements.
- Clean and normalize transaction data.
- Categorize transactions automatically with rules and learned category predictions.
- Allow manual category correction from the UI.
- Show spending dashboards.
- Detect unusual transactions using both model-based and personalized behavior signals.
- Forecast next month expenses.
- Compare forecast models on historical monthly expenses.
- Cluster monthly spending patterns with unsupervised learning.
- Calculate a financial health score.
- Detect recurring payments.
- Generate plain-English financial summaries.
- Answer finance questions through a local query-style chatbot interface.
- Show top suggested finance questions with expandable answers.
- Export categorized transactions to CSV.
- Optionally save transactions to SQLite.
- Optionally use OpenAI for richer AI-generated summaries.

The application is designed to work even without an OpenAI API key. If OpenAI is not configured, it falls back to deterministic local summaries.

## 2. Technology Stack

- Python 3.10+ recommended.
- Streamlit for the web UI.
- pandas for CSV loading, cleaning, and analytics.
- plotly for dashboard charts.
- scikit-learn for anomaly detection, learned categorization, forecasting, forecast evaluation, and clustering.
- SQLite for local lightweight persistence.
- OpenAI Python SDK for optional AI summaries.
- pytest for automated testing.

Note: The project was verified locally with the available Python environment, but the recommended runtime is Python 3.10 or newer.

## 3. Project Structure

```text
.
├── app.py
├── CHALLENGES.md
├── DOCUMENTATION.md
├── README.md
├── requirements.txt
├── data/
│   └── sample_transactions.csv
├── finance/
│   ├── __init__.py
│   ├── analytics.py
│   ├── anomaly_detection.py
│   ├── categorization.py
│   ├── chatbot.py
│   ├── forecasting.py
│   ├── ml_insights.py
│   ├── preprocessing.py
│   └── utils.py
└── tests/
    ├── test_analytics.py
    ├── test_anomaly_detection.py
    ├── test_categorization.py
    ├── test_chatbot.py
    ├── test_ml_insights.py
    └── test_preprocessing.py
```

## 4. Main Application Flow

The main application starts in `app.py`.

High-level flow:

1. Streamlit initializes the page layout.
2. The user uploads a CSV file from the sidebar.
3. The uploaded file is read by `finance.preprocessing.read_csv_safely`.
4. Transactions are cleaned and normalized.
5. Transactions are categorized by `finance.categorization.categorize_transactions`.
6. The user can manually edit categories in the Streamlit transaction editor.
7. Search and filter controls narrow the working dataset.
8. Dashboards are generated from the filtered transactions.
9. AI insights and chatbot responses are generated from the same filtered dataset.
10. The user can export transactions or save them to SQLite.

## 5. File-by-File Explanation

### `app.py`

This is the Streamlit entry point.

Responsibilities:

- Configures the Streamlit page.
- Shows sidebar controls.
- Handles CSV upload.
- Displays the transaction editor.
- Provides search and filter controls.
- Displays dashboard tabs.
- Displays AI insights and chatbot.
- Exports transactions as CSV.
- Saves transactions to SQLite.

Important UI sections:

- Sidebar upload panel.
- Transaction editor.
- Search and filter area.
- Dashboard tab.
- Trends tab.
- ML Insights tab.
- Anomalies tab.
- Recurring payments tab.
- AI insights, top questions, and chatbot section.

### `finance/preprocessing.py`

Handles CSV reading, cleaning, and normalization.

Main functions:

- `read_csv_safely(file)`
- `preprocess_transactions(df, warnings=None)`
- `extract_merchant(description)`

Supported date column names include:

- `date`
- `transaction date`
- `posted date`
- `posting date`
- `value date`

Supported description column names include:

- `description`
- `details`
- `narration`
- `memo`
- `merchant`
- `payee`

Supported amount formats:

- A single `amount` column.
- Separate `debit` and `credit` columns.
- Currency symbols such as `₹`, `$`, and `€`.
- Comma-formatted amounts.
- Negative values in parentheses.

Output columns:

- `date`
- `description`
- `amount`
- `merchant`
- `transaction_type`
- `month`
- `year`
- `day`

Error handling:

- Empty CSV files raise a readable processing error.
- Malformed CSV files raise a readable processing error.
- Missing required amount information raises a readable processing error.
- Invalid rows are dropped with warnings instead of crashing the app.

### `finance/categorization.py`

Categorizes transactions using keyword rules and lightweight learned predictions.

Main functions:

- `categorize_transactions(df)`
- `categorize_transaction(description, amount=0)`
- `apply_manual_categories(df, corrections)`

Supported categories:

- Food
- Travel
- Shopping
- Bills
- Entertainment
- Healthcare
- Salary
- Investments
- Other

Example keyword behavior:

- `Swiggy`, `Zomato`, `restaurant`, `grocery` -> Food
- `Uber`, `Ola`, `flight`, `metro`, `fuel` -> Travel
- `Amazon`, `Flipkart`, `store`, `mall` -> Shopping
- `rent`, `electric`, `internet`, `insurance` -> Bills
- `Netflix`, `Spotify`, `movie`, `cinema` -> Entertainment
- `pharmacy`, `hospital`, `doctor` -> Healthcare
- `salary`, `payroll`, `wages` -> Salary
- `SIP`, `mutual fund`, `stock`, `broker` -> Investments

Learned categorization behavior:

- Existing labeled categories are treated as user training examples when present.
- Confident rule-based labels seed the model when no manual labels exist.
- Transaction descriptions are converted into TF-IDF text features.
- Logistic regression predicts categories for transactions that rules would otherwise mark as `Other`.
- Learned labels are applied only above a confidence threshold.
- The app keeps `category_source` and `category_confidence` columns so users can see whether a label came from rules, existing labels, or learned prediction.

### `finance/analytics.py`

Contains core financial calculations.

Main functions:

- `add_expense_columns(df)`
- `monthly_summary(df)`
- `category_breakdown(df)`
- `top_merchants(df, limit=10)`
- `income_vs_expenses(df)`
- `detect_recurring_expenses(df, min_occurrences=2)`
- `financial_health_score(df)`
- `build_kpis(df)`

Metrics calculated:

- Total income.
- Total expenses.
- Net savings.
- Savings rate.
- Monthly income.
- Monthly expenses.
- Category-wise spending.
- Top merchants.
- Recurring expenses.
- Financial health score.

Financial health score inputs:

- Average savings rate.
- Expense stability.
- Number of months with income.
- Whether latest month had positive savings.

The score is capped between `0` and `100`.

### `finance/anomaly_detection.py`

Detects unusual expenses.

Main function:

- `detect_anomalies(df, contamination=0.05)`

Behavior:

- For larger datasets, it uses scikit-learn `IsolationForest`.
- Adds personalized behavior signals before model scoring.
- For very small datasets, it uses a robust median-based fallback.

Features used by the model:

- Expense amount.
- Day of month.
- Expense compared with category average.
- Merchant frequency.
- Expense compared with usual category amount.
- Expense compared with usual merchant amount.
- Late-night timing when transaction timestamps include time information.

Personalized anomaly reasons include:

- Category deviation, for example spending much more than usual in `Food`.
- Merchant deviation, for example paying much more than usual to the same merchant.
- New merchant, where the merchant has not appeared before in the uploaded dataset.
- Late-night transaction, when timestamp data shows activity between midnight and early morning.

Returned data:

- Only transactions marked as anomalous.
- Includes `expense`, `income`, `anomaly_score`, `is_anomaly`, personalized feature columns, and `anomaly_reason`.

### `finance/forecasting.py`

Predicts next month expenses.

Main functions:

- `forecast_next_month_expenses(df)`
- `evaluate_forecast_models(df, min_train_months=3)`

Next-month forecast behavior:

- If no data exists, returns a zero forecast.
- If fewer than 3 months exist, uses average monthly expenses.
- If 3 or more months exist, uses linear regression blended with the latest 3-month rolling average.

Returned fields:

- `next_month`
- `forecast`
- `method`

Forecast evaluation behavior:

- Uses walk-forward validation after at least 3 training months.
- Compares average, moving average, linear regression, and random forest regressors.
- Reports mean absolute error (`mae`), mean absolute percentage error (`mape`), and evaluated month count.
- Sorts models by lowest error so the best historical method appears first.

### `finance/ml_insights.py`

Creates unsupervised ML insights from monthly spending behavior.

Main functions:

- `cluster_monthly_spending_patterns(df, max_clusters=3)`
- `summarize_spending_clusters(clusters)`

Behavior:

- Converts transactions into monthly category expense vectors.
- Uses category spending shares so months are compared by spending pattern, not only total size.
- Scales features with `StandardScaler`.
- Clusters months with scikit-learn `KMeans`.
- Labels each month with a profile such as `Bills-heavy month`, `Travel-heavy month`, or `mixed-spending month`.

Returned fields include:

- `month`
- `cluster`
- `profile`
- `total_expenses`
- `dominant_category`
- `dominant_category_share`

### `finance/chatbot.py`

Generates AI-style summaries, recommendations, and chatbot answers.

Main functions:

- `generate_financial_insights(df, use_openai=True)`
- `generate_local_summary(df)`
- `generate_recommendations(df)`
- `answer_question(df, question)`
- `TOP_QUESTIONS`

Supported chatbot examples:

- `Where did I spend the most?`
- `How much did I spend on food?`
- `What unusual expenses occurred?`
- `What subscriptions do I have?`
- `How much did I save?`
- `What happened last month?`
- `Show my top 5 expenses`
- `Show Amazon transactions above 1000`
- `Which merchants did I spend the most at?`
- `Compare food spending across months`
- `What should I do to save more?`

OpenAI behavior:

- Uses OpenAI only when `OPENAI_API_KEY` exists and the sidebar toggle is enabled.
- If OpenAI fails for any reason, the app silently falls back to the local summary.

### `finance/utils.py`

Contains shared utility functions.

Main functions:

- `configure_logging()`
- `ensure_data_dir()`
- `get_connection(db_path=DB_PATH)`
- `save_transactions(df, db_path=DB_PATH)`
- `load_transactions(db_path=DB_PATH)`
- `dataframe_to_csv_bytes(df)`
- `currency(value)`

SQLite behavior:

- Database path: `data/finance.db`
- Table name: `transactions`
- The table is created automatically when needed.

## 6. CSV Input Requirements

The app is flexible about column names, but every CSV must contain:

- A date-like column.
- A transaction description-like column, recommended but not strictly required.
- Either an amount column or debit/credit columns.

### Single Amount Column Format

```csv
Date,Description,Amount
2026-01-01,Salary Payroll,120000
2026-01-02,Apartment Rent,-35000
2026-01-03,Swiggy Food Order,-850
```

Rules:

- Positive amounts are treated as income.
- Negative amounts are treated as expenses.

### Debit and Credit Format

```csv
Date,Narration,Debit,Credit
2026-01-02,Rent Bill,35000,
2026-01-01,Salary Payroll,,120000
```

Rules:

- Debit values become negative expenses.
- Credit values become positive income.

### Supported Date Formats

pandas is used for date parsing, with a fallback that also tries day-first parsing.

Examples that should work:

- `2026-01-31`
- `31-01-2026`
- `01/31/2026`
- `31/01/2026`
- `Jan 31 2026`

Rows with dates that still cannot be parsed are dropped with a warning.

## 7. Setup Guide

### Step 1: Open the Project Directory

```bash
cd "/Users/khushalsethi/Personal Finance Assistant"
```

### Step 2: Create a Virtual Environment

```bash
python3 -m venv .venv
```

### Step 3: Activate the Virtual Environment

macOS or Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the App

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

If port `8501` is already in use, run:

```bash
streamlit run app.py --server.port 8502
```

Then open:

```text
http://localhost:8502
```

## 8. Quick Run Guide

From a fresh terminal:

```bash
cd "/Users/khushalsethi/Personal Finance Assistant"
source .venv/bin/activate
streamlit run app.py
```

To test with the included sample file:

1. Open the Streamlit URL in your browser.
2. Use the sidebar CSV uploader.
3. Select `data/sample_transactions.csv`.
4. Review the transaction table.
5. Edit categories if needed.
6. View dashboards and AI insights.
7. Export categorized transactions if desired.

## 9. OpenAI API Setup

The app works without OpenAI, but richer AI summaries can be enabled.

Set your API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Optional model override:

```bash
export OPENAI_MODEL="gpt-4o-mini"
```

Then run:

```bash
streamlit run app.py
```

In the Streamlit sidebar, enable:

```text
Use OpenAI summaries
```

If the API key is missing, invalid, or the OpenAI request fails, the app still uses the local deterministic summary.

## 10. Testing Guide

Run all tests:

```bash
pytest
```

Or explicitly through the virtual environment:

```bash
.venv/bin/python -m pytest
```

Current test coverage includes:

- CSV preprocessing.
- Debit/credit normalization.
- Missing amount validation.
- Keyword and learned categorization.
- Monthly summary calculations.
- Category breakdown calculations.
- Financial health score bounds.
- Small-dataset anomaly detection.
- Personalized anomaly reasons.
- Chatbot intent routing and row retrieval.
- ML spending pattern clustering.
- Forecast model evaluation.

## 11. Manual Workflow Test

Use this checklist after making changes:

1. Start the app with `streamlit run app.py`.
2. Upload `data/sample_transactions.csv`.
3. Confirm transactions appear in the editor.
4. Change one transaction category manually.
5. Search for a merchant such as `Netflix`.
6. Filter by one category.
7. Confirm dashboard metrics update.
8. Open the Trends tab.
9. Open the ML Insights tab.
10. Confirm spending clusters or the minimum-data message appears.
11. Confirm forecast model comparison or the minimum-data message appears.
12. Open the Anomalies tab.
13. Open the Recurring tab.
14. Expand a top question in the chatbot section.
15. Ask: `Where did I spend the most?`
16. Ask: `Show my top 5 expenses`
17. Ask: `What unusual expenses occurred?`
18. Export the categorized CSV.
19. Click the SQLite save button.
20. Toggle `Load saved transactions` and confirm saved transactions load.

## 12. SQLite Persistence

SQLite persistence is optional.

How it works:

- The app creates `data/finance.db`.
- Transactions are saved when the user clicks `Save current transactions to SQLite`.
- Saved data can be loaded by enabling `Load saved transactions`.

Important notes:

- The current implementation appends saved rows.
- It does not de-duplicate transactions.
- It is intended as lightweight local storage, not production-grade multi-user storage.

## 13. Exporting Data

The app provides an export button:

```text
Export filtered transactions as CSV
```

The exported file is named:

```text
categorized_transactions.csv
```

It includes edited categories and normalized transaction fields.

## 14. Logging

Logging is configured by `finance.utils.configure_logging`.

Default log level:

```text
INFO
```

Override with:

```bash
export LOG_LEVEL=DEBUG
```

Useful log scenarios:

- CSV parsing failures.
- Unexpected upload processing errors.
- General runtime diagnostics.

## 15. Error Handling Behavior

The app is designed to avoid crashing on invalid uploads.

Handled cases:

- Empty CSV.
- Malformed CSV.
- Missing date column.
- Missing amount/debit/credit columns.
- Invalid date rows.
- Invalid amount rows.
- OpenAI API failure.
- Empty filtered dataset.

The user receives readable Streamlit messages instead of raw stack traces for expected CSV errors.

## 16. Performance Notes

The app uses Streamlit caching for uploaded CSV processing:

```python
@st.cache_data(show_spinner=False)
def process_upload(uploaded_file):
    ...
```

This avoids reprocessing the same upload during normal Streamlit reruns.

For large CSV files:

- pandas handles the main parsing.
- Invalid rows are dropped after normalization.
- Analytics are grouped by month, category, and merchant.

For very large datasets, future improvements could include chunked CSV loading, database-backed filtering, or pre-aggregated analytics tables.

## 17. Known Limitations

- PDF bank statements are not supported yet.
- OCR is not implemented.
- Multi-user login is not implemented.
- SQLite saves append rows and do not currently de-duplicate.
- Learned categorization improves only when useful labeled examples exist.
- Forecasting is intentionally lightweight, though model comparison is included.
- ML clustering needs at least two months of expense data to produce profiles.
- Forecast model comparison needs at least four months of data.
- Late-night anomaly detection requires CSV timestamps with actual time information, not date-only statements.
- Investment tracking is limited to category detection.
- The dark mode toggle uses lightweight CSS and does not replace Streamlit's native theme system.

## 18. Troubleshooting

### `streamlit: command not found`

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

### `ModuleNotFoundError: No module named 'pandas'`

Dependencies are not installed in the active environment.

Run:

```bash
pip install -r requirements.txt
```

### Port Already in Use

If Streamlit says port `8501` is already in use:

```bash
streamlit run app.py --server.port 8502
```

### CSV Upload Fails

Check that the CSV contains:

- A date column.
- A description or narration column.
- An amount column, or debit and credit columns.

Try the sample file first:

```text
data/sample_transactions.csv
```

### OpenAI Summary Does Not Appear

Confirm:

```bash
echo $OPENAI_API_KEY
```

Then confirm the Streamlit sidebar toggle is enabled:

```text
Use OpenAI summaries
```

If OpenAI still fails, the app will continue using local summaries.

### Tests Cannot Find Modules

Run tests from the project root:

```bash
cd "/Users/khushalsethi/Personal Finance Assistant"
.venv/bin/python -m pytest
```

## 19. Development Guidelines

When extending the app:

- Keep logic inside `finance/` modules, not directly in `app.py`.
- Keep Streamlit-specific code in `app.py`.
- Add focused tests for any new calculation logic.
- Keep CSV parsing permissive and user-friendly.
- Avoid crashing on malformed user uploads.
- Preserve local fallback behavior for AI features.
- Keep features readable before adding complex abstractions.

## 20. Suggested Future Enhancements

High-value next steps:

- PDF statement parsing.
- OCR support for scanned statements.
- User-defined category rules.
- Transaction de-duplication for SQLite.
- Budget targets by category.
- Better recurring payment detection.
- Monthly PDF report export.
- Multi-user login.
- Investment portfolio tracking.
- Model-assisted categorization from user corrections.
- Better mobile layout tuning.

## 21. Command Reference

Create environment:

```bash
python3 -m venv .venv
```

Activate environment:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run app:

```bash
streamlit run app.py
```

Run app on alternate port:

```bash
streamlit run app.py --server.port 8502
```

Run tests:

```bash
pytest
```

Run tests through venv Python:

```bash
.venv/bin/python -m pytest
```

Set OpenAI key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Set OpenAI model:

```bash
export OPENAI_MODEL="gpt-4o-mini"
```

Set debug logging:

```bash
export LOG_LEVEL=DEBUG
```

## 22. Expected End-to-End User Journey

1. User opens the Streamlit app.
2. User uploads a CSV statement.
3. App normalizes the transaction data.
4. App categorizes transactions.
5. User reviews and edits categories.
6. User filters transactions by category, month, or search term.
7. User checks dashboard KPIs.
8. User reviews category and merchant charts.
9. User reviews monthly trends.
10. User reviews anomaly detection.
11. User reviews recurring payments.
12. User reads AI-generated or local financial insights.
13. User asks chatbot questions.
14. User exports the cleaned categorized transactions.
15. User optionally saves transactions to SQLite.

## 23. Verification Status

The project has been verified with:

```bash
.venv/bin/python -m pytest
```

Expected result:

```text
9 passed
```

The Streamlit app was also smoke-checked through a local HTTP request and returned:

```text
HTTP/1.1 200 OK
```
