import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
import pickle as pickle

def read_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    features = ['open', 'high', 'low', 'close', 'volume', 'rsi_3', 'stoch_3_6_slowk', 'stochrsi_3_6_fastk', 'mom_3',
                'willr_3', 'obv_0', 'bbands_3_upperband', 'bbands_3_lowerband', 'ema_3', 'sma_3']
    df_all = df[features].copy()
    df_all["open_close_diff"] = df_all["open"] - df["close"].shift(1)
    df_all.fillna(1, inplace=True)
    df_all["open_close"] = df_all["open_close_diff"].apply(lambda x: 1 if x >= 0 else 0)
    return df_all

file_path = 'Resources/GOOG.US_W1_cleaned.csv'
data = read_and_preprocess(file_path)

# Create a time series using the row number instead of the datetime column
data['day_of_year'] = data.index + 1

# Select relevant columns for time series analysis
time_series_data = data[['day_of_year', 'close']]

# Decompose the time series
decomposition = seasonal_decompose(time_series_data['close'], model='multiplicative', period=12)

# Plot the decomposed components
decomposition.plot()
plt.show()

# Calculate Simple Moving Average (SMA) and Exponential Moving Average (EMA)
data['SMA_20'] = data['close'].rolling(window=20).mean()
data['EMA_20'] = data['close'].ewm(span=20, adjust=False).mean()

# Calculate Volatility (Rolling Standard Deviation)
data['volatility_20'] = data['close'].rolling(window=20).std()

# Calculate Daily Returns
data['daily_return'] = data['close'].pct_change()

# Plot SMA, EMA, and Volatility
plt.figure(figsize=(14, 8))

# Plot SMA and EMA
plt.subplot(3, 1, 1)
plt.plot(data['day_of_year'], data['close'], label='Close Price', color='blue')
plt.plot(data['day_of_year'], data['SMA_20'], label='SMA 20', color='orange')
plt.plot(data['day_of_year'], data['EMA_20'], label='EMA 20', color='green')
plt.title('Close Price with SMA and EMA')
plt.legend()
plt.grid(True)

# Plot Volatility
plt.subplot(3, 1, 2)
plt.plot(data['day_of_year'], data['volatility_20'], label='Volatility (20 days)', color='red')
plt.title('Volatility (Rolling Std Dev)')
plt.legend()
plt.grid(True)

# Plot Daily Returns
plt.subplot(3, 1, 3)
plt.plot(data['day_of_year'], data['daily_return'], label='Daily Return', color='purple')
plt.title('Daily Returns')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Prepare the data for ARIMA and SARIMA models
arima_data = data.dropna(subset=['daily_return'])[['day_of_year', 'close']]

# Split the data into training and test sets
train_data, test_data = train_test_split(arima_data, test_size=0.2, shuffle=False)

# Define and fit the ARIMA model
arima_model = ARIMA(train_data['close'], order=(5, 1, 0))  # ARIMA(p,d,q) where p=5, d=1, q=0
arima_result = arima_model.fit()

# Forecast the next 30 days on the test set
arima_forecast = arima_result.forecast(steps=len(test_data))
arima_mse = mean_squared_error(test_data['close'], arima_forecast)
arima_mae = mean_absolute_error(test_data['close'], arima_forecast)

# Plot the ARIMA forecast
plt.figure(figsize=(12, 6))
plt.plot(train_data['day_of_year'], train_data['close'], label='Train Close Price')
plt.plot(test_data['day_of_year'], test_data['close'], label='Test Close Price')
plt.plot(test_data['day_of_year'], arima_forecast, label='ARIMA Forecast', color='red')
plt.title('ARIMA Model Forecast')
plt.xlabel('Day of the Year')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()

# Define and fit the SARIMA model
sarima_model = SARIMAX(train_data['close'], order=(5, 1, 0), seasonal_order=(1, 1, 1, 12))  # SARIMA(p,d,q)(P,D,Q,s)
sarima_result = sarima_model.fit()
sarima_model_path = 'sarima_model_Monthly.pkl'

# Save the model to a file
with open(sarima_model_path, 'wb') as file:
    pickle.dump(sarima_result, file)

print(f'Model saved to {sarima_model_path}')

# Forecast the next 30 days on the test set
sarima_forecast = sarima_result.get_forecast(steps=len(test_data))
sarima_forecast_mean = sarima_forecast.predicted_mean
sarima_mse = mean_squared_error(test_data['close'], sarima_forecast_mean)
sarima_mae = mean_absolute_error(test_data['close'], sarima_forecast_mean)

# Plot the SARIMA forecast
plt.figure(figsize=(12, 6))
plt.plot(train_data['day_of_year'], train_data['close'], label='Train Close Price')
plt.plot(test_data['day_of_year'], test_data['close'], label='Test Close Price')
plt.plot(test_data['day_of_year'], sarima_forecast_mean, label='SARIMA Forecast', color='red')
plt.title('SARIMA Model Forecast')
plt.xlabel('Day of the Year')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()

# Print model summaries and error metrics
print("ARIMA Model Summary:")
print(arima_result.summary())
print(f"ARIMA Model Mean Squared Error: {arima_mse}")
print(f"ARIMA Model Mean Absolute Error: {arima_mae}")

print("\nSARIMA Model Summary:")
print(sarima_result.summary())
print(f"SARIMA Model Mean Squared Error: {sarima_mse}")
print(f"SARIMA Model Mean Absolute Error: {sarima_mae}")
