
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

#load clean data
df = pd.read_csv('clean_ecommerce_data.csv')

#create Prophet model
model = Prophet()
model.fit(df)

#Create dataframe for future dates
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

#Display forecasted data on graph
fig = model.plot(forecast)
plt.title('Revenue Forecast')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.show()



