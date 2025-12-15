import sqlite3
from datetime import timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import streamlit as st


# =====================================================
# LOAD DATABASE FROM UPLOADED FILE
# =====================================================
@st.cache_data(show_spinner=False)
def load_tables_from_upload(uploaded_file):
    with open("uploaded_cashew.sql", "wb") as f:
        f.write(uploaded_file.getbuffer())

    conn = sqlite3.connect("uploaded_cashew.sql")
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    )["name"].tolist()
    dfs = {t: pd.read_sql(f'SELECT * FROM "{t}";', conn) for t in tables}
    conn.close()
    return dfs


# =====================================================
# SALARY + FINANCIAL MONTHS
# =====================================================
def identify_salary_events(df, threshold):
    salary = df[(df["signed_amount"] > 0) & (df["signed_amount"] >= threshold)].copy()
    salary["salary_date"] = salary["date_created"].dt.normalize()
    salary = salary.sort_values("date_created").drop_duplicates("salary_date")
    return salary["salary_date"]


def assign_financial_months(df, salary_dates):
    if salary_dates.empty:
        df["financial_month_label"] = "UNASSIGNED"
        df["days_since_salary"] = np.nan
        return df

    salary_dates = salary_dates.sort_values().reset_index(drop=True)
    boundaries = []

    for i, start in enumerate(salary_dates):
        end = (
            salary_dates.iloc[i + 1] - timedelta(days=1)
            if i < len(salary_dates) - 1
            else start + pd.DateOffset(months=1)
        )
        boundaries.append((start, end))

    def label(ts):
        for start, end in boundaries:
            if start <= ts <= end:
                return f"{start.strftime('%Y-%m-%d')} ({start.strftime('%b')}-{end.strftime('%b')})"
        return "UNASSIGNED"

    df["financial_month_label"] = df["date_created"].apply(label)
    df["days_since_salary"] = df.apply(
        lambda r: (r["date_created"].normalize()
                   - pd.to_datetime(r["financial_month_label"][:10])).days
        if r["financial_month_label"] != "UNASSIGNED" else np.nan,
        axis=1,
    )
    return df


# =====================================================
# APP SETUP
# =====================================================
st.set_page_config(page_title="Cashew Finance Dashboard", layout="wide")
st.title("Cashew – Personal Finance Dashboard")

uploaded_file = st.sidebar.file_uploader(
    "Upload Cashew .sql export",
    type=["sql"],
)

if uploaded_file is None:
    st.info("Upload a Cashew .sql file to begin.")
    st.stop()


# =====================================================
# LOAD DATA
# =====================================================
dfs = load_tables_from_upload(uploaded_file)

tx = dfs["transactions"].copy()
categories = dfs.get("categories", pd.DataFrame())
wallets = dfs.get("wallets", pd.DataFrame())

tx["date_created"] = pd.to_datetime(tx["date_created"], unit="s", errors="coerce")
tx["signed_amount"] = tx["amount"].astype(float)

if {"category_pk", "name"}.issubset(categories.columns):
    tx = tx.merge(
        categories[["category_pk", "name"]].rename(columns={"name": "category"}),
        left_on="category_fk",
        right_on="category_pk",
        how="left",
    )
else:
    tx["category"] = "UNKNOWN"

if {"wallet_pk", "name"}.issubset(wallets.columns):
    tx = tx.merge(
        wallets[["wallet_pk", "name"]].rename(columns={"name": "wallet"}),
        left_on="wallet_fk",
        right_on="wallet_pk",
        how="left",
    )

tx["wallet"] = tx["wallet"].fillna("UNKNOWN").astype(str)
tx = tx.sort_values("date_created")


# =====================================================
# FILTERS
# =====================================================
st.sidebar.header("Filters")

date_range = st.sidebar.date_input(
    "Date range",
    value=(tx["date_created"].min().date(), tx["date_created"].max().date()),
)

salary_threshold = st.sidebar.number_input(
    "Salary threshold",
    value=float(tx.loc[tx["signed_amount"] > 0, "signed_amount"].quantile(0.9)),
    step=100.0,
)

mask = (
    (tx["date_created"].dt.date >= date_range[0])
    & (tx["date_created"].dt.date <= date_range[1])
)

df = tx.loc[mask].copy()
salary_dates = identify_salary_events(df, salary_threshold)
df = assign_financial_months(df, salary_dates)

df_exp = df[df["signed_amount"] < 0].copy()
df_exp["category_lc"] = df_exp["category"].astype(str).str.lower()
df_exp = df_exp[~df_exp["category_lc"].isin({"jama", "balance correction"})]


# =====================================================
# SIMPLE OVERVIEW
# =====================================================
st.header("Simple overview")

st.subheader("Transactions timeline")

fig, ax = plt.subplots(figsize=(16, 5))
colors = df["signed_amount"].apply(lambda x: "green" if x > 0 else "red")
ax.bar(df["date_created"], df["signed_amount"], color=colors, width=1)
ax.axhline(0, color="black")

for d in salary_dates:
    ax.axvline(d, color="black", linestyle="--")

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("Top spends by financial month")

fm = st.selectbox(
    "Select financial month",
    sorted(df_exp["financial_month_label"].unique()),
)

top = (
    df_exp[df_exp["financial_month_label"] == fm]
    .groupby("category")["signed_amount"]
    .sum()
    .abs()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(8, 5))
top.sort_values().plot(kind="barh", ax=ax, color="red")
st.pyplot(fig)


# =====================================================
# ADVANCED DIAGNOSTICS
# =====================================================
st.header("Advanced diagnostics")

st.subheader("Spend velocity within financial month")
st.caption(
    "Cumulative expenses vs days since salary. "
    "Example: a steep decline in the first week indicates front-loaded spending."
)

fig, ax = plt.subplots(figsize=(10, 5))
for fm, g in df_exp.groupby("financial_month_label"):
    g2 = g.sort_values("days_since_salary")
    ax.plot(g2["days_since_salary"], g2["signed_amount"].cumsum(), alpha=0.6)
st.pyplot(fig)

st.subheader("Expense Pareto curve (80/20)")
st.caption(
    "How to read it step-by-step:\n"
    "• Start at the leftmost point → single largest spending category\n"
    "• Move right → each new category adds less incremental spend\n"
    "• Find where the curve crosses 0.8 (80%)\n"
    "• Categories to the left explain most spending; those to the right have diminishing impact"
)

total_cat = (
    df_exp.groupby("category")["signed_amount"]
    .sum()
    .abs()
    .sort_values(ascending=False)
)
cum = total_cat.cumsum() / total_cat.sum()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(cum.values)
ax.axhline(0.8, linestyle="--")
ax.set_xlabel("Categories (sorted by spend)")
ax.set_ylabel("Cumulative share of spend")
st.pyplot(fig)

st.subheader("Anomaly detection (month-relative)")
st.caption(
    "Z-scores measure deviation from historical behavior. "
    "Example: z > +2 indicates unusually high spending for that category this month; "
    "z < −2 indicates unusually low or suppressed spending."
)

monthly = (
    df_exp.groupby(["financial_month_label", "category"])["signed_amount"]
    .sum()
    .unstack(fill_value=0)
)

latest_month = monthly.index.sort_values()[-1]
z = (monthly.loc[latest_month] - monthly.mean()) / monthly.std().replace(0, np.nan)

fig, ax = plt.subplots(figsize=(8, 5))
z.sort_values().tail(10).plot(kind="barh", ax=ax, color="red")
st.pyplot(fig)
