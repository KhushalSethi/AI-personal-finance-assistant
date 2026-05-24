# Project Challenges and Solutions

This file summarizes the main implementation challenges in the AI Personal Finance Assistant and how the project handles them.

## Messy Bank CSV Formats

Bank exports do not use one standard schema. Some files use `Amount`, others split values into `Debit` and `Credit`, and date or description columns often have different names.

Solution:

- Added flexible column matching in preprocessing.
- Supported common date, description, amount, debit, and credit column names.
- Added currency and comma cleanup for money values.
- Dropped invalid rows with warnings instead of crashing the app.

## Distinguishing Income and Expenses

Financial analytics only work if transaction direction is correct. A single sign mistake can make every expense look like income.

Solution:

- Standardized the internal convention: income is positive and expenses are negative.
- Converted separate debit and credit columns into signed amounts.
- Built analytics helpers that derive `income` and `expense` columns consistently from the signed amount.

## Categorizing Transactions Without a Large Training Dataset

A fully supervised ML categorizer needs labeled transaction examples, which most users do not have at the start.

Solution:

- Kept transparent keyword rules for obvious merchant and description patterns.
- Added learned categorization using TF-IDF and logistic regression when labeled examples are available.
- Used existing corrected categories as training examples.
- Applied learned predictions only when confidence is high enough.
- Exposed `category_source` and `category_confidence` so users can understand whether a category came from rules, existing labels, or learned prediction.

## Making Anomaly Detection Personal

Generic anomaly detection can flag mathematically unusual rows without explaining why they matter to a specific user.

Solution:

- Kept `IsolationForest` for model-based anomaly scoring.
- Added personalized features:
  - merchant frequency
  - category spending ratio
  - merchant spending ratio
  - late-night transaction timing when timestamps include time
- Added human-readable `anomaly_reason` values such as new merchant, category deviation, merchant deviation, and late-night transaction.
- Kept a robust median-based fallback for very small datasets.

## Forecasting With Limited Data

Many users only upload a few months of transactions, which makes complex forecasting unreliable.

Solution:

- Used average monthly expenses for very small histories.
- Used linear regression blended with a rolling average once enough months exist.
- Added walk-forward forecast model evaluation for average, moving average, linear regression, and random forest models.
- Displayed model errors instead of pretending one model is always best.

## Adding ML Without Overcomplicating the App

The project needed stronger ML concepts, but not a heavy training pipeline, vector database, or external API requirement.

Solution:

- Used scikit-learn models already available in the project.
- Added KMeans monthly spending-pattern clustering.
- Added learned categorization and personalized anomaly features as small local modules.
- Avoided requiring OpenAI or any paid API for core functionality.
- Kept model outputs explainable in the UI.

## Making the Chatbot Useful Without an API Key

A fully flexible natural-language agent would usually need an LLM, but the app should still work locally.

Solution:

- Built a local query-style chatbot with intent routing.
- Added support for common finance questions: top categories, top merchants, row retrieval, savings advice, recurring payments, anomalies, forecasts, and comparisons.
- Added top suggested questions in the UI with expandable answers.
- Kept OpenAI optional only for richer summaries.

## Keeping Generated Answers Readable

Generated answers can become hard to read when they include retrieval traces, source chunks, page references, or raw evidence text. That kind of debug output is useful during development, but it hurts the user experience.

Solution:

- Changed local summaries to structured Markdown sections.
- Rendered summaries and chatbot answers with Markdown in Streamlit.
- Tightened the OpenAI prompt so it asks for only final user-facing content.
- Added a cleanup layer that strips `Relevant evidence`, `Sources`, page markers, and chunk IDs before text is displayed.
- Added tests so retrieval/debug text does not leak back into generated answers.

## Keeping the Project Testable

Finance apps can silently produce wrong answers if calculations are not tested.

Solution:

- Added focused tests for preprocessing, categorization, analytics, anomaly detection, chatbot routing, ML clustering, and forecast evaluation.
- Kept core logic inside `finance/` modules so it can be tested without running Streamlit.
- Used deterministic model settings such as fixed random seeds for repeatable tests.
