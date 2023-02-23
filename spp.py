import pandas as pd
import matplotlib.dates as mdates
import plotly.express as px
print('plotly library imported...')
import datetime as dt
import time

import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping
print('tensorflow library imported...')
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
print('sklearn library imported...')

class StockPricePredictor:
    def __init__(self) -> None:
        pass

    def read_data(self, filename, index_col='Date'):
        print('in read')
        try:
            print('Reading dataset...')
            return pd.read_csv(filename, index_col=index_col)
        except Exception as e:
            print(e)
    
    def spp_plot(self, data, template='plotly_dark'):
        fig = px.line(data, template=template)
        fig.show()
    
    def spp_scaler(self, data, scaler):
        scaled_data = scaler.fit_transform(data.values)
        scaled_data = pd.DataFrame(columns=data.columns, data=scaled_data, index=data.index)
        print(f'\nX_feat after transforming to Dataframe -> \n{scaled_data.head()}')
        return scaled_data

    def spp_lstm_split(self, data, steps=10):
        X, y = [], []
        for i in range(len(data)- steps - 1):
            X.append(data[i:i + steps, :-1])
            y.append(data[i + steps-1, -1])

        return np.array(X), np.array(y)

    def spp_data_split(self, x_data, y_data, date_idx, split=0.8):
        split_idx = int(np.ceil(len(x_data) * split))
        date_index = date_idx

        X_train, X_test = x_data[:split_idx], x_data[split_idx:]
        y_train, y_test = y_data[:split_idx], y_data[split_idx:]
        X_train_date, X_test_date = date_index[:split_idx], date_index[split_idx]

        print(f'\n### After Splitting by {split} ###\n X1 Shape: {x_data.shape}\n X_train Shape: {X_train.shape}\n X_test Shape: {X_test.shape}\n y_test Shape: {y_test.shape}')
        return X_train, X_test, X_train_date, X_test_date, y_train, y_test
    
    def spp_lstm_init(self, training_data, model, lstm_cells, dense_cells, optimizer='adam', activation='relu', loss='mean_squared_error', return_sequences=True):
        print('\nLong-Short-Term-Memory (LSTMS) was set to [Sequential]...')
        print('\nAdding layers:')
        model.add(LSTM(lstm_cells, input_shape=(training_data.shape[1], training_data.shape[2]), activation=activation, return_sequences=return_sequences))
        print('LSTM layer with 32 cells (activation = [ReLu]) were added...')
        model.add(Dense(dense_cells))
        print('Dense layer with 1 cell was added...')
        model.compile(loss=loss, optimizer=optimizer)
        print('\nLSTM model was compiled with loss set to [mean_squared_error] and Optimizer set to [Adam]...')
        print('LSTM model summary:')
        model.summary()
        return model

    def spp_fit(self, model, x_train_data, y_train_data, x_test_data, epochs=100, batch_size=4, verbose='auto', shuffle=False):
        print('\nInitialising the fitting/training process...')
        history = model.fit(x_train_data, y_train_data, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=shuffle)
        print('\nInitialising the prediction process...')
        y_pred = model.predict(x_test_data)
        return history, y_pred

def main():
    spp = StockPricePredictor()
    stock_data = spp.read_data('./NFLX.csv')
    print(f'\nraw dataset head: \n{stock_data.head()}')

    target_y = stock_data['Close']
    print('target was sorted based on [Close] cloumn header...')
    X_feat = stock_data.iloc[:, 0:3]
    print(f'\nX_feat after cropped iloc [:, 0:3] -> \n{X_feat.head()}')

    # Presenting the raw data
    # spp.spp_plot(X_feat)

    # Feature Scaling
    print('X_feat will be transformed based on StandardScaler...other option is MinMaxScaler')
    X_ft = spp.spp_scaler(X_feat, StandardScaler())

    X1, y1 = spp.spp_lstm_split(X_ft.values)
    print(f'\nX1:\n{X1[:3]}')
    print(f'\ny1:\n{y1[:3]}')

    X_train, X_test, X_train_date, X_test_date, y_train, y_test = spp.spp_data_split(X1, y1, X_ft.index)
    # lstm_model = spp.spp_lstm_init(X_train, Sequential(), lstm_cells=32, dense_cells=1)

    # history, prediction = spp.spp_fit(lstm_model, X_train, y_train, X_test)
    # timestamp = time.strftime("%Y%m%d-%H%M%S")
    # lstm_model.save(f'saved_models/model-{timestamp}')
    loaded_model = load_model('saved_models/model-20230223-165603')
    loaded_model.summary()
    prediction = loaded_model.predict(X_test)
    print(type(prediction), prediction.shape)
    print(type(X_test), X_test.shape)
    data = pd.DataFrame(data=X_test[0])

    fig = px.line(data)
    fig.show()

    # print(history, prediction)
    # rmse = mean_squared_error(y_test, y_pred, squared=False)
    # mape = mean_absolute_percentage_error(y_test, y_pred)
    # print(f'RMSE = {rmse}')
    # print(f'MAPE = {mape}')

if __name__ == "__main__":
    main()

