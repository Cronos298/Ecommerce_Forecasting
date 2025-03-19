import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly


st.set_page_config(page_title="E-Commerce Revenue Projection Dashboard", layout="wide")

# Load Data
st.title("E-Commerce Revenue Projection")

try:
    df = pd.read_csv('clean_ecommerce_data.csv')  
except FileNotFoundError:
    st.error("Processed data file not found. Please run `etl_pipeline.py` first.")
    st.stop()

# Show Data Preview
df_display = df.rename(columns={'ds': 'Date', 'y': 'Revenue'})

st.subheader("Quick Look at revenue data")
st.dataframe(df_display.head())

# Prophet Forecasting
st.header(" Predicting Future Revenue")

m = Prophet()
m.fit(df)  

# Create DataFrame for next 30 days
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

# Plot Time and Sales Data
st.subheader(" Revenue Trends Over Time")
fig = px.line(forecast, x='ds', y='yhat', labels={'ds': 'Date', 'yhat': 'Projected Revenue'})
fig.update_layout(
    title="Projected Revenue Trends",
    xaxis_title="Date",
    yaxis_title="Revenue (£)",
    template="plotly_white"
)

fig.add_trace(go.Scatter(
    x=df['ds'], y=df['y'],
    mode='markers',
    marker=dict(color='white', size=5, symbol='circle'),
    name='Recorded revenue'
))

st.plotly_chart(fig)

# Plot the Forecast
fig_forecast = plot_plotly(m, forecast)
st.plotly_chart(fig_forecast, use_container_width=True)

# Display Forecast Data
forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)

# Rename columns to more human-friendly names
forecast_display.rename(columns={
    'ds': 'Date',
    'yhat': 'Expected Revenue',
    'yhat_lower': 'Minimum Projection',
    'yhat_upper': 'Maximum Projection'
}, inplace=True)

forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')

# Format values to remove decimals and add £ symbol
forecast_display['Expected Revenue'] = forecast_display['Expected Revenue'].apply(lambda x: f"£{int(round(x))}")
forecast_display['Minimum Projection'] = forecast_display['Minimum Projection'].apply(lambda x: f"£{int(round(x))}")
forecast_display['Maximum Projection'] = forecast_display['Maximum Projection'].apply(lambda x: f"£{int(round(x))}")

# Display the updated table
st.subheader(" Revenue Forecast Summary")
st.dataframe(forecast_display)
