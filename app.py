from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from finance.analytics import (
    build_kpis,
    category_breakdown,
    detect_recurring_expenses,
    financial_health_score,
    income_vs_expenses,
    monthly_summary,
    top_merchants,
)
from finance.anomaly_detection import detect_anomalies
from finance.categorization import CATEGORIES, categorize_transactions
from finance.chatbot import TOP_QUESTIONS, answer_question, generate_financial_insights
from finance.forecasting import evaluate_forecast_models, forecast_next_month_expenses
from finance.ml_insights import cluster_monthly_spending_patterns, summarize_spending_clusters
from finance.preprocessing import CSVProcessingError, read_csv_safely
from finance.utils import configure_logging, currency, dataframe_to_csv_bytes, load_transactions, save_transactions


configure_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Personal Finance Assistant", page_icon="💰", layout="wide")


@st.cache_data(show_spinner=False)
def process_upload(uploaded_file) -> tuple[pd.DataFrame, list[str]]:
    result = read_csv_safely(uploaded_file)
    return categorize_transactions(result.transactions), result.warnings


def apply_theme(mode: str) -> None:
    if mode == "Dark":
        st.markdown(
            """
            <style>
            .stApp { background-color: #111827; color: #f9fafb; }
            [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: #f9fafb; }
            </style>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.sidebar.title("Finance Assistant")
    theme = st.sidebar.radio("Theme", ["Light", "Dark"], horizontal=True)
    apply_theme(theme)

    st.title("AI Personal Finance Assistant")
    st.caption("Upload a CSV bank statement to clean, categorize, analyze, and summarize transactions.")

    uploaded_file = st.sidebar.file_uploader("Upload CSV bank statement", type=["csv"])
    use_openai = st.sidebar.toggle("Use OpenAI summaries", value=False)
    load_saved = st.sidebar.toggle("Load saved transactions", value=False)

    df = pd.DataFrame()
    warnings: list[str] = []

    if uploaded_file:
        try:
            df, warnings = process_upload(uploaded_file)
        except CSVProcessingError as exc:
            st.error(str(exc))
            logger.warning("CSV processing failed: %s", exc)
            return
        except Exception as exc:
            st.error("The file could not be processed. Please check the CSV format and try again.")
            logger.exception("Unexpected upload processing failure: %s", exc)
            return
    elif load_saved:
        df = load_transactions()

    if df.empty:
        show_empty_state()
        return

    for warning in warnings:
        st.warning(warning)

    edited_df = show_transaction_editor(df)
    filtered_df = show_filters(edited_df)

    if st.sidebar.button("Save current transactions to SQLite"):
        saved = save_transactions(edited_df)
        st.sidebar.success(f"Saved {saved} transactions.")

    show_dashboard(filtered_df)
    show_ai_tab(filtered_df, use_openai)


def show_empty_state() -> None:
    st.info("Upload a CSV statement from the sidebar to begin.")
    st.markdown(
        """
        Required fields are flexible. The app understands common names like `Date`,
        `Description`, `Amount`, or separate `Debit` and `Credit` columns.
        """
    )


def show_transaction_editor(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Transactions")
    display = df.copy()
    display["date"] = pd.to_datetime(display["date"]).dt.date
    edited = st.data_editor(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "category": st.column_config.SelectboxColumn("Category", options=CATEGORIES, required=True),
            "amount": st.column_config.NumberColumn("Amount", format="%.2f"),
        },
        disabled=["date", "description", "amount", "merchant", "transaction_type", "month", "year", "day"],
    )
    edited["date"] = pd.to_datetime(edited["date"])
    st.download_button(
        "Export filtered transactions as CSV",
        dataframe_to_csv_bytes(edited),
        file_name="categorized_transactions.csv",
        mime="text/csv",
    )
    return edited


def show_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Search and Filter")
    col1, col2, col3 = st.columns(3)
    with col1:
        search = st.text_input("Search description or merchant")
    with col2:
        categories = st.multiselect("Categories", CATEGORIES, default=CATEGORIES)
    with col3:
        months = st.multiselect("Months", sorted(df["month"].unique()), default=sorted(df["month"].unique()))

    filtered = df.copy()
    if search:
        mask = (
            filtered["description"].str.contains(search, case=False, na=False)
            | filtered["merchant"].str.contains(search, case=False, na=False)
        )
        filtered = filtered[mask]
    if categories:
        filtered = filtered[filtered["category"].isin(categories)]
    if months:
        filtered = filtered[filtered["month"].isin(months)]
    st.caption(f"Showing {len(filtered)} of {len(df)} transactions.")
    return filtered


def show_dashboard(df: pd.DataFrame) -> None:
    tab_dashboard, tab_trends, tab_ml, tab_anomalies, tab_recurring = st.tabs(
        ["Dashboard", "Trends", "ML Insights", "Anomalies", "Recurring"]
    )

    with tab_dashboard:
        kpis = build_kpis(df)
        score, details = financial_health_score(df)
        forecast = forecast_next_month_expenses(df)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Income", currency(kpis["income"]))
        c2.metric("Expenses", currency(kpis["expenses"]))
        c3.metric("Savings", currency(kpis["savings"]))
        c4.metric("Savings Rate", f"{kpis['savings_rate']:.1f}%")
        c5.metric("Health Score", f"{score}/100")

        left, right = st.columns(2)
        categories = category_breakdown(df)
        merchants = top_merchants(df)
        with left:
            if not categories.empty:
                st.plotly_chart(px.pie(categories, names="category", values="total", title="Category Breakdown"), use_container_width=True)
        with right:
            if not merchants.empty:
                st.plotly_chart(px.bar(merchants, x="merchant", y="total", title="Top Merchants"), use_container_width=True)

        st.info(
            f"Next month forecast: {currency(float(forecast['forecast']))} for {forecast['next_month']} "
            f"using {forecast['method'].replace('_', ' ')}. Expense stability: {details['expense_stability']:.1f}%."
        )

    with tab_trends:
        monthly = monthly_summary(df)
        income_expense = income_vs_expenses(df)
        if not monthly.empty:
            st.plotly_chart(
                px.line(monthly, x="month", y=["income", "expenses", "net_savings"], markers=True, title="Monthly Trend"),
                use_container_width=True,
            )
            st.plotly_chart(
                px.bar(income_expense, x="month", y=["income", "expenses"], barmode="group", title="Income vs Expenses"),
                use_container_width=True,
            )
            st.dataframe(monthly, use_container_width=True, hide_index=True)

    with tab_ml:
        clusters = cluster_monthly_spending_patterns(df)
        st.subheader("Spending Pattern Clusters")
        if clusters.empty:
            st.info("At least two months of expense data are needed for clustering.")
        else:
            st.write(summarize_spending_clusters(clusters))
            st.plotly_chart(
                px.bar(
                    clusters,
                    x="month",
                    y="total_expenses",
                    color="profile",
                    title="Monthly Spending Pattern Clusters",
                ),
                use_container_width=True,
            )
            st.dataframe(clusters, use_container_width=True, hide_index=True)

        st.subheader("Forecast Model Evaluation")
        forecast_scores = evaluate_forecast_models(df)
        if forecast_scores.empty:
            st.info("At least four months of data are needed to compare forecast models.")
        else:
            best = forecast_scores.iloc[0]
            st.success(
                f"Best historical forecast model: {best['model'].replace('_', ' ')} "
                f"with MAE {currency(float(best['mae']))}."
            )
            st.dataframe(forecast_scores, use_container_width=True, hide_index=True)

    with tab_anomalies:
        anomalies = detect_anomalies(df)
        if anomalies.empty:
            st.success("No unusual spending detected.")
        else:
            st.warning(f"Detected {len(anomalies)} unusual transaction(s).")
            st.dataframe(anomalies, use_container_width=True, hide_index=True)

    with tab_recurring:
        recurring = detect_recurring_expenses(df)
        if recurring.empty:
            st.info("No recurring payments found yet.")
        else:
            st.dataframe(recurring, use_container_width=True, hide_index=True)


def show_ai_tab(df: pd.DataFrame, use_openai: bool) -> None:
    st.subheader("AI Insights")
    insights = generate_financial_insights(df, use_openai=use_openai)
    st.write(insights.summary)
    for recommendation in insights.recommendations:
        st.write(f"- {recommendation}")

    st.subheader("Ask Your Finance Assistant")
    st.caption("Top questions")
    for top_question in TOP_QUESTIONS:
        with st.expander(top_question):
            st.write(answer_question(df, top_question))

    question = st.text_input("Ask a question", placeholder="Where did I spend the most?")
    if question:
        st.write(answer_question(df, question))


if __name__ == "__main__":
    main()
