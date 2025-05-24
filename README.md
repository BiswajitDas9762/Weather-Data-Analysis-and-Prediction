# Weather-Data-Analysis-and-Prediction
Here's a complete outline and starter code for a Weather Data Analysis and Prediction tool using Python. This project involves:

Analyzing historical weather data (e.g., temperature over time)

Using basic time series visualization and regression techniques to predict future trends.
Install these libraries: pip install pandas matplotlib scikit-learn
Use a CSV file with at least two columns: Date and Temperature.
Python Script:  
weather_prediction.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime

# Load dataset
df = pd.read_csv("weather.csv")

# Convert dates
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = (df['Date'] - df['Date'].min()).dt.days

# Plot historical data
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Temperature'], label='Actual Temperature')
plt.title("Historical Temperature Data")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.show()

# Prepare features and labels
X = df[['Day']]
y = df['Temperature']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future
future_days = 30
last_day = df['Day'].max()
future_X = pd.DataFrame({'Day': [last_day + i for i in range(1, future_days + 1)]})
future_dates = [df['Date'].max() + datetime.timedelta(days=i) for i in range(1, future_days + 1)]
future_y = model.predict(future_X)

# Plot predictions
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Temperature'], label='Historical')
plt.plot(future_dates, future_y, label='Predicted', linestyle='--')
plt.title("Temperature Forecast")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.show()

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on test set: {mse:.2f}")
