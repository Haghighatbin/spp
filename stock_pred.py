from pandas import read_csv, to_datetime, DataFrame
from numpy import array, reshape

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=8,6

print('[importing sklearn preprocessing]')
from sklearn.preprocessing import MinMaxScaler

print('[importing keras models and layers]')
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense


print('[reading the raw dataset]')
df=read_csv("NFLX.csv")
print('dataset head:')
print(df.head())
print(f"dataset shape: {df.shape}")
len_rows, len_cols = df.shape[0], df.shape[1]

df["Date"] = to_datetime(df.Date,format="%Y-%m-%d")
df.index=df['Date']

plt.figure(figsize=(8,6))
plt.plot(df["Close"],label='Close Price history')
plt.legend()
plt.grid()
plt.show()

print('sorting the data ascendingly')
data = df.sort_index(ascending=True,axis=0)
print('generating a new dataset consisting of Data and Close columns')
new_dataset = DataFrame(index=range(0,len(df)),columns=['Date','Close'])

for i in range(0,len(data)):
    new_dataset["Date"][i]=data['Date'][i]
    new_dataset["Close"][i]=data["Close"][i]

print('indexing the new dataset with Date')
new_dataset.index = new_dataset.Date
new_dataset.drop("Date",axis=1,inplace=True)

print('generating the final dataset comprising of values')
final_dataset =new_dataset.values
print(f'the final_dataset: {final_dataset} -- Shape: {final_dataset.shape}')

print('defining the range of the training set [80%]')
train_data = final_dataset[0: int(0.8 * len_rows),:]
print('defing the remaining data as the valid set')
valid_data = final_dataset[int(0.8 * len_rows) + 1:,:]

print('normalising the data within the range of 0 to 1')
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_dataset)

x_train_data,y_train_data = [],[]

for i in range(10,len(train_data)):
    x_train_data.append(scaled_data[i-10:i,0])
    y_train_data.append(scaled_data[i,0])
    
x_train_data,y_train_data = array(x_train_data), array(y_train_data)
x_train_data = reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

print('generating LSTM model')
lstm_model = Sequential()
# lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
lstm_model.add(LSTM(units=50,activation='relu',return_sequences=True,input_shape=(x_train_data.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

lstm_model.compile(loss='mean_squared_error',optimizer='adam')
lstm_model.summary()
print('fitting the generated model to the data')
lstm_model.fit(x_train_data,y_train_data,epochs=50,batch_size=4,verbose=2,shuffle=False)

inputs_data = new_dataset[len(new_dataset)-len(valid_data)-10:].values
inputs_data = inputs_data.reshape(-1,1)
inputs_data = scaler.transform(inputs_data)

X_test = []
for i in range(10,inputs_data.shape[0]):
    X_test.append(inputs_data[i-10:i,0])
X_test = array(X_test)

X_test = reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_closing_price = lstm_model.predict(X_test)
predicted_closing_price=scaler.inverse_transform(predicted_closing_price)
print('saving the lstm model')
lstm_model.save("saved_lstm_model.h5")

train_data = new_dataset[:int(len_rows * 0.8)]
valid_data = new_dataset[int(len_rows * 0.8) + 1:]
valid_data['Predictions'] = predicted_closing_price
plt.plot(train_data['Close'],label='Close price: Training set')
plt.plot(valid_data[['Close','Predictions']], label=['Close price: Valid', 'Close price: Predicted'])
plt.legend()
plt.grid()
plt.show()