import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import tempfile
import os

# Page configuration
st.set_page_config(page_title="Cashew Transaction Timeline", layout="wide")


def load_data_from_db(db_path: str) -> pd.DataFrame:
    """Load transactions from uploaded Cashew SQLite database"""

    conn = sqlite3.connect(db_path)

    query = """
    SELECT 
        t.transaction_pk,
        t.name AS transaction_name,
        t.amount,
        t.date_created,
        t.income,
        t.paid,
        t.note,
        c.name AS category_name,
        c.colour AS category_colour,
        w.name AS wallet_name
    FROM transactions t
    LEFT JOIN categories c ON t.category_fk = c.category_pk
    LEFT JOIN wallets w ON t.wallet_fk = w.wallet_pk
    ORDER BY t.date_created DESC
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # --- SAFE datetime handling ---
    df["date"] = pd.to_datetime(df["date_created"], unit="s", errors="coerce")
    df["date_only"] = df["date"].dt.date
    df["month"] = df["date"].dt.strftime("%Y-%m")

    df["type"] = df["income"].map({1: "Income", 0: "Expense"})
    df["amount_formatted"] = df["amount"].map(lambda x: f"{x:,.2f} ₼")

    return df


def main():
    st.title("💰 Cashew Transaction Timeline")

    st.sidebar.header("Database")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Cashew SQLite database",
        type=["sql", "sqlite", "db"]
    )

    if uploaded_file is None:
        st.info("Please upload a Cashew SQLite database file to continue.")
        st.stop()

    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        df = load_data_from_db(tmp_path)
    except Exception as e:
        st.error("Failed to read database.")
        st.exception(e)
        os.unlink(tmp_path)
        st.stop()

    os.unlink(tmp_path)  # cleanup temp file

    if df.empty:
        st.warning("Database loaded, but no transactions found.")
        st.stop()

    # --------------------
    # SIDEBAR FILTERS
    # --------------------
    st.sidebar.header("Filters")

    min_date = df["date_only"].min()
    max_date = df["date_only"].max()

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    transaction_types = st.sidebar.multiselect(
        "Transaction Type",
        ["Income", "Expense"],
        default=["Income", "Expense"]
    )

    categories = ["All"] + sorted(df["category_name"].dropna().unique())
    wallets = ["All"] + sorted(df["wallet_name"].dropna().unique())

    selected_category = st.sidebar.selectbox("Category", categories)
    selected_wallet = st.sidebar.selectbox("Wallet", wallets)

    # --------------------
    # APPLY FILTERS (NO TIMESTAMP ARITHMETIC)
    # --------------------
    filtered_df = df.copy()

    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df["date_only"] >= start_date) &
            (filtered_df["date_only"] <= end_date)
        ]

    filtered_df = filtered_df[filtered_df["type"].isin(transaction_types)]

    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["category_name"] == selected_category]

    if selected_wallet != "All":
        filtered_df = filtered_df[filtered_df["wallet_name"] == selected_wallet]

    # --------------------
    # SUMMARY METRICS
    # --------------------
    col1, col2, col3, col4 = st.columns(4)

    total_income = filtered_df.loc[filtered_df["income"] == 1, "amount"].sum()
    total_expense = filtered_df.loc[filtered_df["income"] == 0, "amount"].sum()
    net_balance = filtered_df["amount"].sum()  # Sum all amounts (expenses are already negative)

    col1.metric("Transactions", len(filtered_df))
    col2.metric("Income", f"{total_income:,.2f} ₼")
    col3.metric("Expenses", f"{total_expense:,.2f} ₼")
    col4.metric("Net", f"{net_balance:,.2f} ₼")

    # --------------------
    # TABS
    # --------------------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📅 Timeline",
        "📊 Analytics",
        "🔍 Deep Dive",
        "📋 Transactions",
        "📈 Monthly",
        "🔮 Predictive",
        "📉 Time-Series",
        "🔗 Correlations",
        "💸 Money Flow"
    ])

    # --------------------
    # TIMELINE
    # --------------------
    with tab1:
        if filtered_df.empty:
            st.info("No data for selected filters.")
        else:
            fig = go.Figure()

            for t, c in [("Income", "green"), ("Expense", "red")]:
                d = filtered_df[filtered_df["type"] == t]
                fig.add_bar(
                    x=d["date"],
                    y=d["amount"],
                    name=t,
                    marker_color=c
                )

            fig.update_layout(
                height=500,
                barmode="group",
                xaxis_title="Date",
                yaxis_title="Amount (₼)"
            )

            st.plotly_chart(fig, use_container_width=True)

    # --------------------
    # ANALYTICS TAB
    # --------------------
    with tab2:
        st.header("📊 Spending Patterns & Analytics")
        
        # Spending Patterns Section
        st.subheader("📅 Spending Patterns")
        
        expense_df = filtered_df[filtered_df["income"] == 0].copy()
        
        if not expense_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Day of week analysis
                expense_df["day_of_week"] = expense_df["date"].dt.day_name()
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                dow_spending = (
                    expense_df.groupby("day_of_week")["amount"]
                    .sum()
                    .abs()
                    .reindex(day_order)
                    .reset_index()
                )
                dow_spending.columns = ["Day", "Amount"]
                
                fig_dow = px.bar(
                    dow_spending,
                    x="Day",
                    y="Amount",
                    title="Spending by Day of Week",
                    color="Amount",
                    color_continuous_scale="Reds"
                )
                st.plotly_chart(fig_dow, use_container_width=True)
            
            with col2:
                # Average transaction size by day
                avg_by_day = (
                    expense_df.groupby("day_of_week")["amount"]
                    .mean()
                    .abs()
                    .reindex(day_order)
                    .reset_index()
                )
                avg_by_day.columns = ["Day", "Avg Amount"]
                
                fig_avg = px.bar(
                    avg_by_day,
                    x="Day",
                    y="Avg Amount",
                    title="Average Transaction Size by Day",
                    color="Avg Amount",
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig_avg, use_container_width=True)
            
            # Weekday vs Weekend
            col3, col4 = st.columns(2)
            
            with col3:
                expense_df["is_weekend"] = expense_df["date"].dt.dayofweek >= 5
                weekend_comparison = (
                    expense_df.groupby("is_weekend")["amount"]
                    .sum()
                    .abs()
                    .reset_index()
                )
                weekend_comparison["Day Type"] = weekend_comparison["is_weekend"].map({
                    True: "Weekend", False: "Weekday"
                })
                
                fig_weekend = px.pie(
                    weekend_comparison,
                    values="amount",
                    names="Day Type",
                    title="Weekday vs Weekend Spending",
                    hole=0.4,
                    color_discrete_sequence=["lightblue", "darkblue"]
                )
                st.plotly_chart(fig_weekend, use_container_width=True)
        else:
            st.info("No expense data to analyze.")
        
        st.divider()
        
        # Spending Velocity Section
        st.subheader("⚡ Spending Velocity (Post-Salary)")
        
        # Get salary transactions
        salary_df = df[
            df["transaction_name"].str.lower().str.contains("salary", na=False) |
            df["category_name"].str.lower().str.contains("salary", na=False)
        ].sort_values("date")
        
        if len(salary_df) >= 1:
            velocity_data = []
            
            for _, salary_row in salary_df.iterrows():
                salary_date = salary_row["date"]
                
                # Get expenses in next 30 days after salary
                post_salary = df[
                    (df["date"] > salary_date) &
                    (df["date"] <= salary_date + pd.Timedelta(days=30)) &
                    (df["income"] == 0)
                ].copy()
                
                if not post_salary.empty:
                    post_salary["days_after_salary"] = (post_salary["date"] - salary_date).dt.days
                    
                    for day in range(1, 31):
                        day_expenses = post_salary[post_salary["days_after_salary"] == day]["amount"].sum()
                        cumulative_expenses = post_salary[post_salary["days_after_salary"] <= day]["amount"].sum()
                        
                        velocity_data.append({
                            "Salary Date": salary_date.strftime("%Y-%m-%d"),
                            "Days After Salary": day,
                            "Daily Expense": abs(day_expenses),
                            "Cumulative Expense": abs(cumulative_expenses)
                        })
            
            if velocity_data:
                velocity_df = pd.DataFrame(velocity_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Cumulative spending curve for each salary period
                    fig_cumulative = px.line(
                        velocity_df,
                        x="Days After Salary",
                        y="Cumulative Expense",
                        color="Salary Date",
                        title="Cumulative Spending After Salary",
                        labels={"Cumulative Expense": "Amount (₼)"},
                        markers=True
                    )
                    st.plotly_chart(fig_cumulative, use_container_width=True)
                
                with col2:
                    # Average daily spending by day after salary
                    avg_velocity = (
                        velocity_df.groupby("Days After Salary")["Daily Expense"]
                        .mean()
                        .reset_index()
                    )
                    
                    fig_avg_velocity = px.bar(
                        avg_velocity,
                        x="Days After Salary",
                        y="Daily Expense",
                        title="Average Daily Spending by Day After Salary",
                        labels={"Daily Expense": "Amount (₼)"}
                    )
                    st.plotly_chart(fig_avg_velocity, use_container_width=True)
                
                # Insights
                week1_avg = velocity_df[velocity_df['Days After Salary'] <= 7]['Daily Expense'].mean()
                week2_avg = velocity_df[(velocity_df['Days After Salary'] > 7) & (velocity_df['Days After Salary'] <= 14)]['Daily Expense'].mean()
                week3_plus_avg = velocity_df[velocity_df['Days After Salary'] > 14]['Daily Expense'].mean()
                
                st.info(f"""
                **💡 Spending Velocity Insights:**
                - First 7 days avg: {week1_avg:.2f} ₼/day
                - Days 8-14 avg: {week2_avg:.2f} ₼/day
                - Days 15-30 avg: {week3_plus_avg:.2f} ₼/day
                """)
        else:
            st.warning("Need salary transactions to calculate spending velocity.")
        
        st.divider()
        
        # Category Heatmap over Months
        st.subheader("📊 Category Spending Heatmap")
        
        expense_data = filtered_df[filtered_df["income"] == 0].copy()
        
        if not expense_data.empty:
            # Create pivot table for heatmap
            heatmap_data = (
                expense_data.groupby(["month", "category_name"])["amount"]
                .sum()
                .abs()
                .reset_index()
            )
            
            heatmap_pivot = heatmap_data.pivot(
                index="category_name",
                columns="month",
                values="amount"
            ).fillna(0)
            
            # Create heatmap
            fig_heatmap = px.imshow(
                heatmap_pivot,
                labels=dict(x="Month", y="Category", color="Amount (₼)"),
                title="Category Spending Over Months",
                color_continuous_scale="Reds",
                aspect="auto",
                zmin=0,
                zmax=1500
            )
            
            fig_heatmap.update_layout(height=600)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("No expense data to display heatmap.")

    # --------------------
    # DEEP DIVE TAB
    # --------------------
    with tab3:
        st.header("🔍 Deep Dive Analysis")
        
        # Category Deep Dive
        st.subheader("📂 Category Deep Dive")
        
        expense_categories = sorted(filtered_df[filtered_df["income"] == 0]["category_name"].dropna().unique())
        
        if expense_categories:
            selected_deep_category = st.selectbox("Select Category to Analyze", expense_categories)
            
            cat_data = filtered_df[
                (filtered_df["category_name"] == selected_deep_category) &
                (filtered_df["income"] == 0)
            ].copy()
            
            if not cat_data.empty:
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Total Spent", f"{abs(cat_data['amount'].sum()):,.2f} ₼")
                col2.metric("Transactions", len(cat_data))
                col3.metric("Avg Transaction", f"{abs(cat_data['amount'].mean()):,.2f} ₼")
                col4.metric("Largest Transaction", f"{cat_data['amount'].abs().max():,.2f} ₼")
                
                # Month-over-month trend
                st.subheader("Monthly Trend")
                cat_monthly = (
                    cat_data.groupby("month")["amount"]
                    .sum()
                    .abs()
                    .reset_index()
                )
                cat_monthly.columns = ["Month", "Amount"]
                
                fig_trend = px.line(
                    cat_monthly,
                    x="Month",
                    y="Amount",
                    title=f"{selected_deep_category} - Monthly Spending Trend",
                    markers=True
                )
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # All transactions sorted by amount
                st.subheader("All Transactions (Sorted)")
                cat_data_copy = cat_data.copy()
                cat_data_copy["amount_abs"] = cat_data_copy["amount"].abs()
                all_transactions = (
                    cat_data_copy.sort_values("amount_abs", ascending=False)[
                        ["date", "transaction_name", "amount_formatted", "wallet_name"]
                    ].copy()
                )
                all_transactions["date"] = all_transactions["date"].dt.strftime("%Y-%m-%d")
                st.dataframe(all_transactions, use_container_width=True, hide_index=True)
        else:
            st.info("No expense categories found.")
        
        st.divider()
        
        # Comparative Analytics
        st.subheader("📊 Comparative Analytics")
        
        comparison_type = st.radio(
            "Compare:",
            ["Current vs Previous Period", "Month over Month", "Category Comparison"],
            horizontal=True
        )
        
        if comparison_type == "Current vs Previous Period":
            # Get salary periods
            salary_df = df[
                df["transaction_name"].str.lower().str.contains("salary", na=False) |
                df["category_name"].str.lower().str.contains("salary", na=False)
            ].sort_values("date")
            
            if len(salary_df) >= 2:
                # Current period (last salary to now)
                current_start = salary_df.iloc[-1]["date"]
                current_end = pd.Timestamp.now()
                current_data = df[(df["date"] >= current_start) & (df["date"] <= current_end)]
                
                # Previous period (second to last salary)
                prev_start = salary_df.iloc[-2]["date"]
                prev_end = salary_df.iloc[-1]["date"] - pd.Timedelta(days=1)
                prev_data = df[(df["date"] >= prev_start) & (df["date"] <= prev_end)]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Current Period")
                    curr_income = current_data[current_data["income"] == 1]["amount"].sum()
                    curr_expense = current_data[current_data["income"] == 0]["amount"].sum()
                    curr_net = current_data["amount"].sum()
                    
                    st.metric("Income", f"{curr_income:,.2f} ₼")
                    st.metric("Expenses", f"{curr_expense:,.2f} ₼")
                    st.metric("Net", f"{curr_net:,.2f} ₼")
                
                with col2:
                    st.markdown("### Previous Period")
                    prev_income = prev_data[prev_data["income"] == 1]["amount"].sum()
                    prev_expense = prev_data[prev_data["income"] == 0]["amount"].sum()
                    prev_net = prev_data["amount"].sum()
                    
                    income_change = ((curr_income - prev_income) / prev_income * 100) if prev_income != 0 else 0
                    expense_change = ((abs(curr_expense) - abs(prev_expense)) / abs(prev_expense) * 100) if prev_expense != 0 else 0
                    net_change = ((curr_net - prev_net) / abs(prev_net) * 100) if prev_net != 0 else 0
                    
                    st.metric("Income", f"{prev_income:,.2f} ₼", delta=f"{income_change:+.1f}%")
                    st.metric("Expenses", f"{prev_expense:,.2f} ₼", delta=f"{expense_change:+.1f}%", delta_color="inverse")
                    st.metric("Net", f"{prev_net:,.2f} ₼", delta=f"{net_change:+.1f}%")
                
                # Category comparison
                st.subheader("Category Changes")
                
                curr_cat = current_data[current_data["income"] == 0].groupby("category_name")["amount"].sum().abs()
                prev_cat = prev_data[prev_data["income"] == 0].groupby("category_name")["amount"].sum().abs()
                
                comparison_df = pd.DataFrame({
                    "Current": curr_cat,
                    "Previous": prev_cat
                }).fillna(0)
                comparison_df["Change"] = comparison_df["Current"] - comparison_df["Previous"]
                comparison_df["Change %"] = (comparison_df["Change"] / comparison_df["Previous"] * 100).replace([float('inf'), -float('inf')], 0)
                comparison_df = comparison_df.sort_values("Change", ascending=False)
                
                fig_comparison = go.Figure()
                fig_comparison.add_trace(go.Bar(name="Previous", x=comparison_df.index, y=comparison_df["Previous"], marker_color="lightblue"))
                fig_comparison.add_trace(go.Bar(name="Current", x=comparison_df.index, y=comparison_df["Current"], marker_color="darkblue"))
                fig_comparison.update_layout(barmode="group", title="Category Spending: Current vs Previous Period")
                st.plotly_chart(fig_comparison, use_container_width=True)
            else:
                st.warning("Need at least 2 salary periods for comparison.")
        
        elif comparison_type == "Month over Month":
            monthly_comparison = (
                filtered_df[filtered_df["income"] == 0]
                .groupby("month")["amount"]
                .sum()
                .abs()
                .reset_index()
            )
            monthly_comparison.columns = ["Month", "Expenses"]
            monthly_comparison["Previous Month"] = monthly_comparison["Expenses"].shift(1)
            monthly_comparison["Change"] = monthly_comparison["Expenses"] - monthly_comparison["Previous Month"]
            monthly_comparison["Change %"] = (monthly_comparison["Change"] / monthly_comparison["Previous Month"] * 100).round(1)
            
            st.dataframe(
                monthly_comparison.style.format({
                    "Expenses": "{:,.2f} ₼",
                    "Previous Month": "{:,.2f} ₼",
                    "Change": "{:+,.2f} ₼",
                    "Change %": "{:+.1f}%"
                }),
                use_container_width=True,
                hide_index=True
            )
        
        else:  # Category Comparison
            category_totals = (
                filtered_df[filtered_df["income"] == 0]
                .groupby("category_name")["amount"]
                .agg(["sum", "count", "mean"])
                .abs()
                .sort_values("sum", ascending=False)
                .reset_index()
            )
            category_totals.columns = ["Category", "Total", "Count", "Average"]
            
            st.dataframe(
                category_totals.style.format({
                    "Total": "{:,.2f} ₼",
                    "Count": "{:,.0f}",
                    "Average": "{:,.2f} ₼"
                }),
                use_container_width=True,
                hide_index=True
            )

    # --------------------
    # TRANSACTIONS TAB
    # --------------------
    with tab4:
        st.dataframe(
            filtered_df[
                ["date", "transaction_name", "amount_formatted",
                 "type", "category_name", "wallet_name", "note"]
            ],
            use_container_width=True,
            height=400
        )

    # --------------------
    # MONTHLY TAB
    # --------------------
    with tab5:
        st.subheader("Calendar Monthly Summary")
        
        monthly = (
            filtered_df
            .groupby(["month", "type"])["amount"]
            .sum()
            .reset_index()
        )

        pivot = monthly.pivot(index="month", columns="type", values="amount").fillna(0)

        if {"Income", "Expense"}.issubset(pivot.columns):
            pivot["Net"] = pivot["Income"] + pivot["Expense"]  # Expenses are already negative

        st.dataframe(pivot.style.format("{:,.2f} ₼"), use_container_width=True)
        
        st.divider()
        st.subheader("Salary-Based Period Analysis")
        
        # Identify salary transactions from the full dataset (not filtered)
        salary_df = df[
            df["transaction_name"].str.lower().str.contains("salary", na=False) |
            df["category_name"].str.lower().str.contains("salary", na=False)
        ].sort_values("date")
        
        if len(salary_df) < 2:
            st.warning("Need at least 2 salary transactions to analyze salary periods.")
        else:
            # Create salary periods
            salary_dates = salary_df["date"].tolist()
            periods = []
            
            for i in range(len(salary_dates) - 1):
                period_start = salary_dates[i]
                period_end = salary_dates[i + 1] - pd.Timedelta(days=1)
                start_month = period_start.strftime('%b')
                end_month = period_end.strftime('%b')
                month_range = f"({start_month}-{end_month})" if start_month != end_month else f"({start_month})"
                period_label = f"{period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')} {month_range}"
                periods.append({
                    "label": period_label,
                    "start": period_start,
                    "end": period_end
                })
            
            # Add period from last salary to today
            if salary_dates:
                last_salary = salary_dates[-1]
                now = pd.Timestamp.now()
                start_month = last_salary.strftime('%b')
                end_month = now.strftime('%b')
                month_range = f"({start_month}-{end_month})" if start_month != end_month else f"({start_month})"
                periods.append({
                    "label": f"{last_salary.strftime('%Y-%m-%d')} to Present {month_range}",
                    "start": last_salary,
                    "end": now
                })
            
            # Period selector
            period_labels = [p["label"] for p in periods]
            selected_period_label = st.selectbox("Select Salary Period", period_labels, index=len(period_labels)-1)
            
            # Get selected period
            selected_period = periods[period_labels.index(selected_period_label)]
            
            # Filter data for selected period
            period_df = df[
                (df["date"] >= selected_period["start"]) &
                (df["date"] <= selected_period["end"])
            ]
            
            # Calculate statistics
            col1, col2, col3 = st.columns(3)
            
            period_income = period_df.loc[period_df["income"] == 1, "amount"].sum()
            period_expense = period_df.loc[period_df["income"] == 0, "amount"].sum()
            period_net = period_df["amount"].sum()
            
            col1.metric("Period Income", f"{period_income:,.2f} ₼")
            col2.metric("Period Expenses", f"{period_expense:,.2f} ₼")
            col3.metric("Period Net", f"{period_net:,.2f} ₼")
            
            # Category breakdown for the period
            st.subheader("Spending by Category")
            
            # Pareto Chart for the selected period
            period_expense_data = period_df[period_df["income"] == 0].copy()
            
            if not period_expense_data.empty:
                # Calculate category totals
                pareto_data = (
                    period_expense_data.groupby("category_name")["amount"]
                    .sum()
                    .abs()
                    .sort_values(ascending=False)
                    .reset_index()
                )
                pareto_data.columns = ["Category", "Amount"]
                
                # Calculate cumulative percentage
                pareto_data["Cumulative Amount"] = pareto_data["Amount"].cumsum()
                total_amount = pareto_data["Amount"].sum()
                pareto_data["Cumulative %"] = (pareto_data["Cumulative Amount"] / total_amount * 100).round(1)
                pareto_data["% of Total"] = (pareto_data["Amount"] / total_amount * 100).round(1)
                
                # Create Pareto chart with dual axis
                fig_pareto = go.Figure()
                
                # Bar chart for amounts
                fig_pareto.add_trace(go.Bar(
                    x=pareto_data["Category"],
                    y=pareto_data["Amount"],
                    name="Amount",
                    marker_color="steelblue",
                    yaxis="y",
                    text=pareto_data["% of Total"].apply(lambda x: f"{x}%"),
                    textposition="outside"
                ))
                
                # Line chart for cumulative percentage
                fig_pareto.add_trace(go.Scatter(
                    x=pareto_data["Category"],
                    y=pareto_data["Cumulative %"],
                    name="Cumulative %",
                    marker_color="red",
                    mode="lines+markers",
                    yaxis="y2",
                    line=dict(width=3)
                ))
                
                # Add 80% reference line
                fig_pareto.add_hline(
                    y=80, 
                    line_dash="dash", 
                    line_color="orange",
                    annotation_text="80%",
                    yref="y2"
                )
                
                # Update layout with dual y-axes
                fig_pareto.update_layout(
                    title=f"Pareto Chart: Category Spending for Selected Period",
                    xaxis_title="Category",
                    yaxis=dict(
                        title="Amount (₼)",
                        side="left"
                    ),
                    yaxis2=dict(
                        title="Cumulative %",
                        side="right",
                        overlaying="y",
                        range=[0, 100]
                    ),
                    height=500,
                    showlegend=True,
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_pareto, use_container_width=True)
                
                # Show which categories make up 80% of spending
                categories_80 = pareto_data[pareto_data["Cumulative %"] <= 80]["Category"].tolist()
                if categories_80:
                    st.info(f"""
                    **💡 Pareto Insight (80/20 Rule):**
                    
                    The top **{len(categories_80)}** categories account for approximately 80% of your spending in this period:
                    
                    {', '.join(categories_80)}
                    """)
            
            st.subheader("Category Breakdown")
            category_breakdown = (
                period_df[period_df["income"] == 0]
                .groupby("category_name")["amount"]
                .sum()
                .abs()
                .sort_values(ascending=False)
                .reset_index()
            )
            category_breakdown.columns = ["Category", "Amount"]
            category_breakdown["Amount_Formatted"] = category_breakdown["Amount"].apply(lambda x: f"{x:,.2f} ₼")
            
            st.dataframe(
                category_breakdown[["Category", "Amount_Formatted"]],
                use_container_width=True,
                hide_index=True
            )
            
            # Daily spending trend
            st.subheader("Daily Spending in Period")
            daily_period = (
                period_df
                .groupby(period_df["date"].dt.date)["amount"]
                .sum()
                .reset_index()
            )
            daily_period.columns = ["Date", "Amount"]
            
            fig_daily = px.line(
                daily_period,
                x="Date",
                y="Amount",
                title="Daily Net Flow",
                markers=True
            )
            fig_daily.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_daily, use_container_width=True)
            
            # Transaction list for the period
            st.subheader("All Transactions in Period")
            period_display = period_df[[
                "date", "transaction_name", "amount_formatted",
                "type", "category_name", "wallet_name"
            ]].sort_values("date", ascending=False).copy()
            period_display["date"] = period_display["date"].dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(period_display, use_container_width=True, height=300, hide_index=True)

    # --------------------
    # PREDICTIVE ANALYTICS TAB
    # --------------------
    with tab6:
        st.header("🔮 Predictive Analytics")
        
        # Expense Forecasting
        st.subheader("📊 Expense Forecasting")
        
        expense_df = filtered_df[filtered_df["income"] == 0].copy()
        
        if not expense_df.empty and len(expense_df["month"].unique()) >= 3:
            # Monthly aggregation
            monthly_expenses = (
                expense_df.groupby("month")["amount"]
                .sum()
                .abs()
                .reset_index()
            )
            monthly_expenses.columns = ["Month", "Amount"]
            monthly_expenses = monthly_expenses.sort_values("Month")
            
            # Simple linear trend for forecast
            from sklearn.linear_model import LinearRegression
            import numpy as np
            
            X = np.arange(len(monthly_expenses)).reshape(-1, 1)
            y = monthly_expenses["Amount"].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Forecast next 3 months
            future_X = np.arange(len(monthly_expenses), len(monthly_expenses) + 3).reshape(-1, 1)
            forecast = model.predict(future_X)
            
            # Create forecast dataframe
            last_month = pd.to_datetime(monthly_expenses["Month"].iloc[-1] + "-01")
            future_months = pd.date_range(start=last_month + pd.DateOffset(months=1), periods=3, freq='MS')
            
            forecast_df = pd.DataFrame({
                "Month": future_months.strftime("%Y-%m"),
                "Forecast": forecast,
                "Type": "Forecast"
            })
            
            monthly_expenses["Type"] = "Actual"
            combined = pd.concat([
                monthly_expenses.rename(columns={"Amount": "Forecast"}),
                forecast_df
            ])
            
            # Plot
            fig_forecast = px.line(
                combined,
                x="Month",
                y="Forecast",
                color="Type",
                title="Expense Forecast (Next 3 Months)",
                markers=True,
                color_discrete_map={"Actual": "blue", "Forecast": "orange"}
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Show predictions
            col1, col2, col3 = st.columns(3)
            col1.metric("Next Month Forecast", f"{forecast[0]:,.2f} ₼")
            col2.metric("Month +2 Forecast", f"{forecast[1]:,.2f} ₼")
            col3.metric("Month +3 Forecast", f"{forecast[2]:,.2f} ₼")
            
            # Average monthly spending
            avg_monthly = monthly_expenses["Amount"].mean()
            st.info(f"**Average Monthly Spending:** {avg_monthly:,.2f} ₼")
        else:
            st.warning("Need at least 3 months of data for forecasting.")
        
        st.divider()
        
        # Budget Burn Rate
        st.subheader("🔥 Budget Burn Rate Analysis")
        
        # Get current salary period
        salary_df = df[
            df["transaction_name"].str.lower().str.contains("salary", na=False) |
            df["category_name"].str.lower().str.contains("salary", na=False)
        ].sort_values("date")
        
        if not salary_df.empty:
            current_salary_date = salary_df.iloc[-1]["date"]
            current_period_df = df[df["date"] >= current_salary_date].copy()
            
            days_elapsed = (pd.Timestamp.now() - current_salary_date).days
            
            current_expense = current_period_df[current_period_df["income"] == 0]["amount"].sum()
            current_income = current_period_df[current_period_df["income"] == 1]["amount"].sum()
            
            # Assume 30-day period
            daily_burn_rate = abs(current_expense) / days_elapsed if days_elapsed > 0 else 0
            projected_monthly = daily_burn_rate * 30
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Days Since Salary", days_elapsed)
            col2.metric("Daily Burn Rate", f"{daily_burn_rate:,.2f} ₼/day")
            col3.metric("Projected Monthly", f"{projected_monthly:,.2f} ₼")
            
            # Burn rate chart
            daily_expenses = (
                current_period_df[current_period_df["income"] == 0]
                .groupby(current_period_df[current_period_df["income"] == 0]["date"].dt.date)["amount"]
                .sum()
                .abs()
                .cumsum()
                .reset_index()
            )
            daily_expenses.columns = ["Date", "Cumulative"]
            
            # Add projected line
            if days_elapsed > 0:
                projection_dates = pd.date_range(
                    start=current_salary_date.date(),
                    periods=30,
                    freq='D'
                )
                projection_amounts = [daily_burn_rate * i for i in range(30)]
                
                fig_burn = go.Figure()
                
                fig_burn.add_trace(go.Scatter(
                    x=daily_expenses["Date"],
                    y=daily_expenses["Cumulative"],
                    name="Actual Spending",
                    mode="lines+markers",
                    line=dict(color="red", width=3)
                ))
                
                fig_burn.add_trace(go.Scatter(
                    x=projection_dates,
                    y=projection_amounts,
                    name="Projected at Current Rate",
                    mode="lines",
                    line=dict(color="orange", dash="dash", width=2)
                ))
                
                fig_burn.update_layout(
                    title="Spending Burn Rate vs Projection",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Spending (₼)",
                    height=400
                )
                
                st.plotly_chart(fig_burn, use_container_width=True)
                
                remaining_income = current_income + current_expense
                days_remaining = remaining_income / daily_burn_rate if daily_burn_rate > 0 else 0
                
                if days_remaining > 0:
                    st.warning(f"⚠️ At current burn rate, your income will last approximately **{days_remaining:.0f} days**")
        else:
            st.info("No salary transactions found for burn rate analysis.")
        
        st.divider()
        
        # Anomaly Detection
        st.subheader("⚠️ Anomaly Detection")
        
        if not expense_df.empty:
            # Calculate z-scores
            expense_amounts = expense_df["amount"].abs()
            mean_amount = expense_amounts.mean()
            std_amount = expense_amounts.std()
            
            expense_df["z_score"] = (expense_amounts - mean_amount) / std_amount
            anomalies = expense_df[expense_df["z_score"].abs() > 2].copy()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Anomalies Detected", len(anomalies))
                st.metric("Mean Transaction", f"{mean_amount:,.2f} ₼")
                st.metric("Std Deviation", f"{std_amount:,.2f} ₼")
            
            with col2:
                # Distribution plot
                fig_dist = px.histogram(
                    expense_amounts,
                    nbins=50,
                    title="Transaction Amount Distribution",
                    labels={"value": "Amount (₼)", "count": "Frequency"}
                )
                fig_dist.add_vline(x=mean_amount, line_dash="dash", line_color="red", annotation_text="Mean")
                fig_dist.add_vline(x=mean_amount + 2*std_amount, line_dash="dot", line_color="orange", annotation_text="+2σ")
                st.plotly_chart(fig_dist, use_container_width=True)
            
            if not anomalies.empty:
                st.subheader("Detected Anomalies (>2 Standard Deviations)")
                anomaly_display = anomalies[[
                    "date", "transaction_name", "amount_formatted", 
                    "category_name", "z_score"
                ]].copy()
                anomaly_display["date"] = anomaly_display["date"].dt.strftime("%Y-%m-%d")
                anomaly_display["z_score"] = anomaly_display["z_score"].round(2)
                anomaly_display = anomaly_display.sort_values("z_score", ascending=False, key=abs)
                
                st.dataframe(anomaly_display, use_container_width=True, hide_index=True)

    # --------------------
    # TIME-SERIES ANALYSIS TAB
    # --------------------
    with tab7:
        st.header("📉 Time-Series Analysis")
        
        expense_df = filtered_df[filtered_df["income"] == 0].copy()
        
        if not expense_df.empty:
            # Daily expenses
            daily_expenses = (
                expense_df.groupby(expense_df["date"].dt.date)["amount"]
                .sum()
                .abs()
                .reset_index()
            )
            daily_expenses.columns = ["Date", "Amount"]
            daily_expenses["Date"] = pd.to_datetime(daily_expenses["Date"])
            daily_expenses = daily_expenses.sort_values("Date")
            
            # Calculate rolling averages
            daily_expenses["7-Day MA"] = daily_expenses["Amount"].rolling(window=7, min_periods=1).mean()
            daily_expenses["30-Day MA"] = daily_expenses["Amount"].rolling(window=30, min_periods=1).mean()
            
            # Rolling averages plot
            st.subheader("📊 Rolling Averages")
            
            fig_rolling = go.Figure()
            
            fig_rolling.add_trace(go.Scatter(
                x=daily_expenses["Date"],
                y=daily_expenses["Amount"],
                name="Daily Spending",
                mode="lines",
                line=dict(color="lightgray", width=1),
                opacity=0.5
            ))
            
            fig_rolling.add_trace(go.Scatter(
                x=daily_expenses["Date"],
                y=daily_expenses["7-Day MA"],
                name="7-Day Moving Average",
                mode="lines",
                line=dict(color="blue", width=2)
            ))
            
            fig_rolling.add_trace(go.Scatter(
                x=daily_expenses["Date"],
                y=daily_expenses["30-Day MA"],
                name="30-Day Moving Average",
                mode="lines",
                line=dict(color="red", width=2)
            ))
            
            fig_rolling.update_layout(
                title="Spending with Moving Averages",
                xaxis_title="Date",
                yaxis_title="Amount (₼)",
                height=500,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig_rolling, use_container_width=True)
            
            st.divider()
            
            # Trend Analysis
            st.subheader("📈 Trend Analysis")
            
            # Linear trend
            from sklearn.linear_model import LinearRegression
            import numpy as np
            
            X = np.arange(len(daily_expenses)).reshape(-1, 1)
            y = daily_expenses["Amount"].values
            
            model = LinearRegression()
            model.fit(X, y)
            trend = model.predict(X)
            
            # Calculate trend direction
            trend_slope = model.coef_[0]
            trend_direction = "📈 Increasing" if trend_slope > 0 else "📉 Decreasing"
            
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Trend Direction", trend_direction)
            col2.metric("Daily Change", f"{trend_slope:,.2f} ₼/day")
            col3.metric("Monthly Impact", f"{trend_slope * 30:,.2f} ₼/month")
            
            # Plot with trend line
            fig_trend = go.Figure()
            
            fig_trend.add_trace(go.Scatter(
                x=daily_expenses["Date"],
                y=daily_expenses["Amount"],
                name="Actual",
                mode="markers",
                marker=dict(size=6, color="blue", opacity=0.6)
            ))
            
            fig_trend.add_trace(go.Scatter(
                x=daily_expenses["Date"],
                y=trend,
                name="Trend Line",
                mode="lines",
                line=dict(color="red", width=3, dash="dash")
            ))
            
            fig_trend.update_layout(
                title="Spending Trend Analysis",
                xaxis_title="Date",
                yaxis_title="Amount (₼)",
                height=400
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
            
            st.divider()
            
            # Volatility Analysis
            st.subheader("📊 Spending Volatility")
            
            # Calculate volatility metrics
            std_dev = daily_expenses["Amount"].std()
            mean_spending = daily_expenses["Amount"].mean()
            coefficient_of_variation = (std_dev / mean_spending) * 100
            
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Standard Deviation", f"{std_dev:,.2f} ₼")
            col2.metric("Mean Daily Spending", f"{mean_spending:,.2f} ₼")
            col3.metric("Coefficient of Variation", f"{coefficient_of_variation:.1f}%")
            
            if coefficient_of_variation < 50:
                st.success("✅ Your spending is relatively consistent")
            elif coefficient_of_variation < 100:
                st.warning("⚠️ Moderate spending volatility detected")
            else:
                st.error("🔴 High spending volatility - consider budgeting")

    # --------------------
    # CORRELATION ANALYSIS TAB
    # --------------------
    with tab8:
        st.header("🔗 Correlation Analysis")
        
        expense_df = filtered_df[filtered_df["income"] == 0].copy()
        
        if not expense_df.empty:
            # Category correlation over time
            st.subheader("📊 Category Spending Correlations")
            
            # Create category spending matrix
            category_monthly = (
                expense_df.groupby(["month", "category_name"])["amount"]
                .sum()
                .abs()
                .reset_index()
            )
            
            category_pivot = category_monthly.pivot(
                index="month",
                columns="category_name",
                values="amount"
            ).fillna(0)
            
            if len(category_pivot.columns) > 1 and len(category_pivot) > 2:
                # Calculate correlation matrix
                correlation_matrix = category_pivot.corr()
                
                # Heatmap
                fig_corr = px.imshow(
                    correlation_matrix,
                    labels=dict(color="Correlation"),
                    title="Category Spending Correlation Matrix",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    aspect="auto"
                )
                
                fig_corr.update_layout(height=600)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Find strong correlations
                st.subheader("🔍 Strong Correlations (|r| > 0.5)")
                
                strong_corr = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_value = correlation_matrix.iloc[i, j]
                        if abs(corr_value) > 0.5:
                            strong_corr.append({
                                "Category 1": correlation_matrix.columns[i],
                                "Category 2": correlation_matrix.columns[j],
                                "Correlation": corr_value
                            })
                
                if strong_corr:
                    corr_df = pd.DataFrame(strong_corr)
                    corr_df = corr_df.sort_values("Correlation", ascending=False, key=abs)
                    st.dataframe(
                        corr_df.style.format({"Correlation": "{:.2f}"}),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No strong correlations found between categories.")
            else:
                st.warning("Need more data for correlation analysis.")
            
            st.divider()
            
            # Day of week correlation
            st.subheader("📅 Day-of-Week Spending Patterns")
            
            expense_df["day_of_week"] = expense_df["date"].dt.dayofweek
            expense_df["week"] = expense_df["date"].dt.isocalendar().week
            
            dow_weekly = (
                expense_df.groupby(["week", "day_of_week"])["amount"]
                .sum()
                .abs()
                .reset_index()
            )
            
            if len(dow_weekly["week"].unique()) > 2:
                dow_pivot = dow_weekly.pivot(
                    index="week",
                    columns="day_of_week",
                    values="amount"
                ).fillna(0)
                
                day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                dow_pivot.columns = [day_names[i] if i < 7 else str(i) for i in dow_pivot.columns]
                
                dow_corr = dow_pivot.corr()
                
                fig_dow_corr = px.imshow(
                    dow_corr,
                    labels=dict(color="Correlation"),
                    title="Day-of-Week Spending Correlation",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1
                )
                
                st.plotly_chart(fig_dow_corr, use_container_width=True)

    # --------------------
    # MONEY FLOW SANKEY TAB
    # --------------------
    with tab9:
        st.header("💸 Money Flow Visualization")
        
        st.subheader("📊 Income → Wallet → Category Flow")
        
        # Prepare data for Sankey diagram
        income_df = filtered_df[filtered_df["income"] == 1].copy()
        expense_df = filtered_df[filtered_df["income"] == 0].copy()
        
        if not income_df.empty and not expense_df.empty:
            # Create flow data
            flows = []
            
            # Income to Wallet
            income_by_wallet = income_df.groupby("wallet_name")["amount"].sum().abs()
            for wallet, amount in income_by_wallet.items():
                flows.append({
                    "source": "Income",
                    "target": f"Wallet: {wallet}",
                    "value": amount
                })
            
            # Wallet to Category
            expense_by_wallet_cat = expense_df.groupby(["wallet_name", "category_name"])["amount"].sum().abs()
            for (wallet, category), amount in expense_by_wallet_cat.items():
                flows.append({
                    "source": f"Wallet: {wallet}",
                    "target": f"Category: {category}",
                    "value": amount
                })
            
            flows_df = pd.DataFrame(flows)
            
            # Create node list
            all_nodes = list(set(flows_df["source"].tolist() + flows_df["target"].tolist()))
            node_dict = {node: idx for idx, node in enumerate(all_nodes)}
            
            # Map to indices
            flows_df["source_idx"] = flows_df["source"].map(node_dict)
            flows_df["target_idx"] = flows_df["target"].map(node_dict)
            
            # Create Sankey diagram
            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_nodes,
                    color=["green" if "Income" in node else "blue" if "Wallet" in node else "red" for node in all_nodes]
                ),
                link=dict(
                    source=flows_df["source_idx"].tolist(),
                    target=flows_df["target_idx"].tolist(),
                    value=flows_df["value"].tolist()
                )
            )])
            
            fig_sankey.update_layout(
                title="Money Flow: Income → Wallets → Categories",
                height=600,
                font=dict(size=10)
            )
            
            st.plotly_chart(fig_sankey, use_container_width=True)
            
            # Summary statistics
            st.subheader("📈 Flow Summary")
            
            col1, col2, col3 = st.columns(3)
            
            total_income = income_by_wallet.sum()
            total_expenses = flows_df[flows_df["target"].str.contains("Category")]["value"].sum()
            retention_rate = ((total_income - total_expenses) / total_income * 100) if total_income > 0 else 0
            
            col1.metric("Total Income Flow", f"{total_income:,.2f} ₼")
            col2.metric("Total Expense Flow", f"{total_expenses:,.2f} ₼")
            col3.metric("Retention Rate", f"{retention_rate:.1f}%")
            
        else:
            st.warning("Need both income and expense data for money flow analysis.")


if __name__ == "__main__":
    main()
