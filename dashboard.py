import streamlit as st
import sqlite3
import pandas as pd
import os

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fraud_saas.db")

# Extract database path for SQLite or use PostgreSQL
if "sqlite" in DATABASE_URL:
    DB_PATH = DATABASE_URL.replace("sqlite:///", "").replace("./", "")
else:
    DB_PATH = DATABASE_URL

st.sidebar.title("⚙️ Controls")
st.sidebar.info("Admin View")
st.set_page_config(
    page_title="Fraud Detection SaaS Dashboard",
    layout="wide"
)

st.title("🛡️ Fraud Detection SaaS – Admin Dashboard")

# ---------------------------------------
# DB connection
# ---------------------------------------
@st.cache_data
def load_data():
    if "sqlite" in DATABASE_URL:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM prediction_logs", conn)
        conn.close()
    else:
        # PostgreSQL support
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL)
        df = pd.read_sql("SELECT * FROM prediction_logs", conn)
        conn.close()
    return df

df = load_data()

if df.empty:
    st.warning("No predictions yet.")
    st.stop()

# ---------------------------------------
# Metrics
# ---------------------------------------
total = len(df)
frauds = df[df["fraud_prediction"] == 1].shape[0]
fraud_rate = (frauds / total) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Total Predictions", total)
col2.metric("Fraud Cases", frauds)
col3.metric("Fraud Rate (%)", f"{fraud_rate:.2f}")

st.divider()

# ---------------------------------------
# Charts
# ---------------------------------------
st.subheader("📊 Fraud Probability Distribution")
st.bar_chart(df["fraud_probability"])

st.subheader("📈 Fraud vs Non-Fraud")
st.bar_chart(df["fraud_prediction"].value_counts())

# ---------------------------------------
# Recent Predictions
# ---------------------------------------
st.subheader("🧾 Recent Transactions")
st.dataframe(
    df.sort_values("created_at", ascending=False).head(20),
    use_container_width=True
)