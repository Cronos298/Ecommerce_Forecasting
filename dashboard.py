import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly

# Set Streamlit page config
st.set_page_config(page_title=" E-Commerce Sales Forecasting Dashboard", layout="wide")

# Load Data
st.title(" E-Commerce Sales Forecasting Dashboard")

try:
    df = pd.read_csv('clean_ecommerce_data.csv')  # Load cleaned dataset
except FileNotFoundError:
    st.error(" Cleaned data file not found. Please run `etl_pipeline.py` first.")
    st.stop()  # Stop execution if data is missing

# Display Data Preview
df_display = df.rename(columns={'ds':'Date', 'y':'Actual Sales'})

st.subheader(" Data Overview")
st.dataframe(df_display.head())



# Prophet Forecasting
st.subheader("🔮 Forecasting Future Sales")

# Train the Prophet Model
m = Prophet()
m.fit(df)  # Train on historical data

# Create Future DataFrame (30 Days into the Future)
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

# Plot Time Sales Data
st.subheader(" Sales Over Time")
fig = px.line(forecast, x='ds', y='yhat', labels={'ds':'Date','yhat':'Predicted Sales'})
fig.update_layout(
    title="Sales Forecast",
    xaxis_title="Date",
    yaxis_title="Sales (£)",
    template="plotly_white"
)

fig.add_trace(go.Scatter(
    x=df['ds'], y=df['y'],
    mode='markers',
    marker=dict(color='white', size=5,symbol='circle'),
    name='Actual Sales'
))

st.plotly_chart(fig)

# Plot the Forecast
fig_forecast = plot_plotly(m, forecast)
st.plotly_chart(fig_forecast, use_container_width=True)

# Display Forecast Data

forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)

# Rename columns to more readable names
forecast_display.rename(columns={
    'ds': 'Date',
    'yhat': 'Predicted Sales',
    'yhat_lower': 'Lower Estimate',
    'yhat_upper': 'Upper Estimate'  # Fixed typo
}, inplace=True)

forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')

# Format values to remove decimals and add £ symbol
forecast_display['Predicted Sales'] = forecast_display['Predicted Sales'].apply(lambda x: f"£{int(round(x))}")
forecast_display['Lower Estimate'] = forecast_display['Lower Estimate'].apply(lambda x: f"£{int(round(x))}")
forecast_display['Upper Estimate'] = forecast_display['Upper Estimate'].apply(lambda x: f"£{int(round(x))}")

# Display the updated table
st.subheader("📊 Forecasted Data")
st.dataframe(forecast_display)

