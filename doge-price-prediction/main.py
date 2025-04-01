import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("DOGE-USD.csv")
data.head()

data.corr(numeric_only=True)
data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True)
data.set_index('Date', inplace=True)
data.isnull().any()

data.isnull().sum()

data = data.dropna()
data.describe()

plt.figure(figsize=(20, 7))
x = data.groupby('Date')['Close'].mean()
x.plot(linewidth=2.5, color='b')
plt.xlabel('Date')
plt.ylabel('Volume')

plt.title('Date vs Close of 2021')
plt.show()

data['gap'] = (data['High'] - data['Low']) * data['Volume']
data['y'] = data['High'] / data['Volume']
data['z'] = data['Low'] / data['Volume']
data['a'] = data['High'] / data['Low']
data['b'] = (data['High'] / data['Low']) * data['Volume']

data = data[['Close', 'Volume', 'gap', 'a', 'b']]
df2 = data.tail(30)
train = df2[:11]
test = df2[-19:]

from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(endog=train['Close'], exog=train.drop('Close', axis=1), order=(2, 1, 1))
results = model.fit()

start = 11
end = 29
predictions = results.predict(start=start, end=end, exog=test.drop('Close', axis=1))
test['Close'].plot(legend=True, figsize=(12, 6), label="Actual Close", color='blue')
predictions.plot(label='Predicted Close', legend=True, color='red')

plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Actual vs Predicted Close Prices')
plt.legend()
plt.show()