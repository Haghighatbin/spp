import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import datetime as dt

import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import *
# from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error
# from sklearn.model_selection import train_test_split, TimeSeriesSplit

stock_data = pd.read_csv('./NFLX.csv', index_col='Date')
print(stock_data.head())

# plt.figure(figsize=(15,10))
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=60))
# x_dates = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in stock_data.index.values]

# plt.plot(x_dates, stock_data['High'], label='High')
# plt.plot(x_dates, stock_data['Low'], label='Low')
# plt.xlabel('Time Scale')
# plt.ylabel('Scaled USD')
# plt.legend()
# plt.gcf().autofmt_xdate()
# plt.show()

target_y = stock_data['Close']
X_feat = stock_data.iloc[:,0:3]

# Feature Scaling
sc = StandardScaler() # MinMaxScaler is another method
X_ft = sc.fit_transform(X_feat.values)
X_ft = pd.DataFrame(columns=X_feat.columns, data=X_ft, index=X_feat.index)
print(X_ft)

def lstm_split(data, n_steps):
    X, y = [], []
    for i in range(len(data)-n_steps+1):
        X.append(data[i:i + n_steps, :-1])
        y.append(data[i + n_steps-1, -1])

    return np.array(X), np.array(y)

X1, y1 = lstm_split(X_ft.values, n_steps=2)
print(f"X1: {X1}")
print(f"y1: {y1}")
# train_split = 0.8
# split_idx = int(np.ceil(len(X1) * train_split))
# date_index = X_ft.index

# X_train, X_test = X1[:split_idx], X1[split_idx:]
# y_train, y_test = y1[:split_idx], y1[split_idx:]
# X_train_date, X_test_date = date_index[:split_idx], date_index[split_idx:]

# print(X1.shape, X_train.shape, X_test.shape, y_test.shape)
# print(X1)

# lstm = Sequential()
# lstm.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=True))
# lstm.add(Dense(1))
# lstm.compile(loss='mean_squared_error', optimizer='adam')
# lstm.summary()

# history = lstm.fit(X_train, y_train, epochs=100, batch_size=4, verbose=2, shuffle=False)
# y_pred = lstm.predict(X_test)
# print(y_pred.shape, y_test.shape)
# # print(y_pred)
# rmse = mean_squared_error(y_test, y_pred, squared=False)
# mape = mean_absolute_percentage_error(y_test, y_pred)
# print(f'RMSE = {rmse}')
# print(f'MAPE = {mape}')

