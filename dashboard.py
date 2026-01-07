import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# ----------------- PAGE SETTINGS -----------------
st.set_page_config(
    page_title="Sales Dashboard",
    layout="wide"
)

# ----------------- LOAD DATA -----------------
df = pd.read_csv("data/sales.csv")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# ----------------- CUSTOM DARK STYLE -----------------
st.markdown("""
    <style>
        .metric-card {
            padding: 18px;
            border-radius: 16px;
            background: #0D0D0D;
            border: 1px solid #262626;
            box-shadow: 0px 6px 20px rgba(0,0,0,0.6);
        }

        /* Main page background */
        .stApp {
            background: #000000;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #050505;
        }

        /* Dataframe background */
        .stDataFrame {
            background: #0D0D0D;
        }
    </style>
""", unsafe_allow_html=True)


st.title("🛒 Walmart Sales Dashboard")
st.caption("Analyze weekly sales, economic impact, and forecasting ")

# ----------------- KPIS -----------------
total_sales = df["Weekly_Sales"].sum()
avg_sales = df["Weekly_Sales"].mean()
top_store = df.groupby("Store")["Weekly_Sales"].sum().idxmax()

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Total Sales</h4>
        <h2>${total_sales:,.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Average Weekly Sales</h4>
        <h2>${avg_sales:,.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Top Performing Store</h4>
        <h2>Store {top_store}</h2>
    </div>
    """, unsafe_allow_html=True)

st.write("---")

# ----------------- SIDEBAR FILTERS -----------------
st.sidebar.header("Filters")

start_date = st.sidebar.date_input("Start Date", df["Date"].min())
end_date = st.sidebar.date_input("End Date", df["Date"].max())

store_filter = st.sidebar.multiselect(
    "Select Store",
    options=df["Store"].unique(),
    default=df["Store"].unique()
)

holiday_filter = st.sidebar.selectbox(
    "Holiday?",
    ["All", "Holiday Only", "Non-Holiday"]
)

filtered_df = df[
    (df["Date"] >= pd.to_datetime(start_date)) &
    (df["Date"] <= pd.to_datetime(end_date)) &
    (df["Store"].isin(store_filter))
]

if holiday_filter == "Holiday Only":
    filtered_df = filtered_df[filtered_df["Holiday_Flag"] == 1]
elif holiday_filter == "Non-Holiday":
    filtered_df = filtered_df[filtered_df["Holiday_Flag"] == 0]

# ----------------- TABS -----------------
tab1, tab2, tab3 = st.tabs(["📈 Trends", "🏬 Store Insights", "🔮 Forecast"])

# ---- TAB 1: TRENDS ----
with tab1:
    st.subheader("Sales Trend Over Time")
    trend = filtered_df.groupby("Date")["Weekly_Sales"].sum().reset_index()
    fig = px.line(trend, x="Date", y="Weekly_Sales", markers=True, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ---- TAB 2: STORE ----
with tab2:
    st.subheader("Sales by Store")
    store_sales = filtered_df.groupby("Store")["Weekly_Sales"].sum().reset_index()
    fig2 = px.bar(store_sales, x="Store", y="Weekly_Sales", template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Raw Data")
    st.dataframe(filtered_df, use_container_width=True)

# ---- TAB 3: FORECAST ----
with tab3:
    st.subheader("Next 8 Weeks Forecast")

    forecast_df = df.groupby("Date")["Weekly_Sales"].sum().reset_index()
    forecast_df["time_index"] = np.arange(len(forecast_df))

    X = forecast_df[["time_index"]]
    y = forecast_df["Weekly_Sales"]

    model = LinearRegression()
    model.fit(X, y)

    future_index = np.arange(len(forecast_df), len(forecast_df) + 8).reshape(-1, 1)
    future_sales = model.predict(future_index)

    future_dates = pd.date_range(start=forecast_df["Date"].max(), periods=8, freq="W")

    forecast_results = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Sales": future_sales
    })

    fig3 = px.line(forecast_results, x="Date", y="Predicted_Sales", markers=True, template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)
