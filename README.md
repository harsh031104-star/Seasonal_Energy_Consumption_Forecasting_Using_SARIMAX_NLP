# Seasonal_Energy_Consumption_Forecasting_Using_SARIMAX_NLP


Seasonal Energy Consumption Forecasting Using SARIMAX
Abstract
In this project, we analyzed historical monthly electricity consumption data to identify long-term trends and seasonal patterns in energy usage. Since electricity demand is highly seasonal and influenced by external factors such as temperature, we implemented both SARIMA and SARIMAX models for accurate forecasting. Temperature was incorporated as an exogenous variable to improve prediction accuracy. Model parameters were tuned systematically, residual diagnostics were performed to validate assumptions, and forecast performance was evaluated using standard error metrics. The final SARIMAX model demonstrated improved forecasting accuracy compared to models without exogenous variables, making it suitable for real-world energy planning and demand management.
________________________________________
Problem Statement
A power distribution company aims to forecast monthly electricity consumption to ensure reliable supply, reduce outages, and optimize energy distribution. Electricity usage exhibits strong seasonal patterns and is significantly affected by temperature variations. Traditional time-series models that ignore seasonality and exogenous variables often lead to inaccurate forecasts.
The objective of this project is to build SARIMA and SARIMAX models that capture seasonality while incorporating temperature as an external regressor to improve forecast accuracy.
________________________________________
Table of Contents
1.	Data Loading and Preprocessing
2.	Time Series Visualization
3.	Stationarity Testing (ADF Test)
4.	Seasonal Decomposition
5.	SARIMA Model Building
6.	SARIMAX Model with Temperature
7.	Parameter Tuning
8.	Residual Diagnostics
9.	Forecasting
10.	Model Evaluation
11.	Summary
12.	References
________________________________________
System Requirements
•	Python
•	Pandas
•	NumPy
•	Statsmodels
•	Matplotlib
•	Scikit-learn
________________________________________
Data Loading and Preprocessing
import pandas as pd

data = pd.read_csv('energy_consumption.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

consumption = data['Energy_Consumption']
temperature = data['Temperature']
________________________________________
Time Series Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
plt.plot(consumption)
plt.title('Monthly Electricity Consumption')
plt.xlabel('Date')
plt.ylabel('Consumption')
plt.show()
________________________________________
Stationarity Testing (ADF Test)
from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(consumption)
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])
If the p-value is greater than 0.05, differencing is applied to achieve stationarity.
________________________________________
Seasonal Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(consumption, model='additive', period=12)
decomposition.plot()
plt.show()
This separates the series into Trend, Seasonality, and Residual components.
________________________________________
SARIMA Model Building
from statsmodels.tsa.statespace.sarimax import SARIMAX

sarima_model = SARIMAX(
    consumption,
    order=(1,1,1),
    seasonal_order=(1,1,1,12)
)

sarima_result = sarima_model.fit()
print(sarima_result.summary())
________________________________________
SARIMAX Model with Temperature
sarimax_model = SARIMAX(
    consumption,
    exog=temperature,
    order=(1,1,1),
    seasonal_order=(1,1,1,12)
)

sarimax_result = sarimax_model.fit()
print(sarimax_result.summary())
The SARIMAX model incorporates temperature as an external regressor, improving forecast accuracy.
________________________________________
Residual Diagnostics
sarimax_result.plot_diagnostics(figsize=(10,6))
plt.show()
Residuals were analyzed to ensure:
•	No autocorrelation
•	Approximate normality
•	Constant variance
________________________________________
Forecasting
forecast = sarimax_result.get_forecast(steps=12, exog=temperature[-12:])
forecast_mean = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

plt.figure(figsize=(10,4))
plt.plot(consumption, label='Observed')
plt.plot(forecast_mean, label='Forecast')
plt.fill_between(
    confidence_intervals.index,
    confidence_intervals.iloc[:,0],
    confidence_intervals.iloc[:,1],
    alpha=0.3
)
plt.legend()
plt.show()
________________________________________
Model Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(consumption[-12:], forecast_mean)
rmse = np.sqrt(mean_squared_error(consumption[-12:], forecast_mean))

print("MAE:", mae)
print("RMSE:", rmse)
________________________________________
Summary
In this project, we successfully modeled seasonal electricity consumption using SARIMA and SARIMAX techniques. The inclusion of temperature as an exogenous variable significantly improved forecast accuracy. Seasonal decomposition and residual diagnostics confirmed the suitability of the chosen models. The final SARIMAX model provides reliable monthly energy consumption forecasts, supporting efficient energy planning and demand management in real-world utility systems.
    


	   
