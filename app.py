import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Page Config
st.set_page_config(page_title="Sales & Growth Predictor", page_icon="📈", layout="wide")

# Synthetic Time-Series Data Generator
@st.cache_data
def load_sales_data():
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=24, freq="ME")
    # Steady upward growth + seasonal holiday peaks + random variation
    trend = np.linspace(12000, 35000, 24)
    seasonality = 3500 * np.sin(np.linspace(0, 4 * np.pi, 24))
    noise = np.random.normal(0, 1200, 24)
    sales = np.maximum(trend + seasonality + noise, 5000)
    return pd.DataFrame({'Date': dates, 'Sales ($)': sales})

df = load_sales_data()

# App Header
st.title("📈 Simple Sales & Revenue Predictor")
st.markdown("Estimate how much money your business will make in the coming months based on past growth patterns.")

# Sidebar Controls
st.sidebar.header("Prediction Controls")
forecast_months = st.sidebar.slider(
    "How many months ahead do you want to project?", 
    min_value=1, 
    max_value=12, 
    value=6,
    help="Choose the timeline for your future estimates."
)

# Convert dates into simple step numbers (1, 2, 3...) to calculate average growth
df['Month_Number'] = np.arange(1, len(df) + 1)

X = df[['Month_Number']]
y = df['Sales ($)']

# Fit trendline (line of best fit)
model = LinearRegression()
model.fit(X, y)

# Calculate monthly growth speed
monthly_growth = model.coef_[0]

# Calculate future projections
future_months = np.arange(len(df) + 1, len(df) + 1 + forecast_months).reshape(-1, 1)
future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.DateOffset(months=1), periods=forecast_months, freq="ME")
future_sales = model.predict(future_months)

future_df = pd.DataFrame({'Date': future_dates, 'Sales ($)': future_sales, 'Type': 'Future Estimate'})
df['Type'] = 'Past Sales'

combined_df = pd.concat([df[['Date', 'Sales ($)', 'Type']], future_df])

# Easy-to-understand KPI Cards
col1, col2, col3 = st.columns(3)
col1.metric("Expected Sales Next Month", f"${future_sales[0]:,.0f}", help="Estimated income for the very next month.")
col2.metric("Average Monthly Growth", f"+${monthly_growth:,.0f} / month", help="On average, how much your sales increase every single month.")
col3.metric(f"Total Projected Revenue ({forecast_months} Months)", f"${future_sales.sum():,.0f}", help="Sum of all predicted monthly revenues combined.")

# Chart Visual
st.subheader("Past Performance vs. Expected Future Growth")

fig = px.line(
    combined_df, 
    x='Date', 
    y='Sales ($)', 
    color='Type', 
    markers=True, 
    template='plotly_white',
    color_discrete_map={'Past Sales': '#1f77b4', 'Future Estimate': '#ff7f0e'},
    labels={'Sales ($)': 'Monthly Revenue ($)', 'Date': 'Month', 'Type': 'Legend'}
)
st.plotly_chart(fig, use_container_width=True)

# Plain-English Summary Box
st.info(
    f"💡 **How to read this chart:**\n\n"
    f"* **Blue Line:** Your real recorded sales history over the last 2 years.\n"
    f"* **Orange Line:** The future estimate. Based on overall patterns, sales are steadily rising by roughly **${monthly_growth:,.0f} each month**."
)