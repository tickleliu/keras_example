from pandas import Series, DataFrame
import pandas as pd
import numpy as np

df = pd.read_csv('jena_climate_2009_2016.csv', sep=',')
print(df.columns)

data = []
for col in df.columns:
    if col == 'Date Time':
        continue
    else:
        item = df[col]
        data.append(item)

data = np.asarray(data, dtype=np.float32)
data = np.transpose(data, [1, 0])

from matplotlib import pyplot as plt

data_shape = data.shape
# plt.plot(range(data_shape[0]), data[:, 1])
mean = data.mean(axis=0)
std = data.std(axis=0)
data -= mean
data /= std

lookback = 720  # i.e. our observations will go back 5 days
steps = 6  # i.e. our observations will be sampled at one data point per hour.
delay = 144  # i.e. our targets will be 24 hour in the future.

# train test data
x_train = []
y_train = []
batch_size = 1000
for i in range(batch_size):
    sample = data[i: i + lookback, :]
    sample = np.array(sample)
    sample = sample[0:-1: steps, :]
    label = data[i + lookback + delay, 1]
    x_train.append(sample)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape)
print(y_train.shape)

x_test = []
y_test = []
test_offset = 100000
for j in range(batch_size):
    i = j + test_offset
    sample = data[i: i + lookback, :]
    sample = np.array(sample)
    sample = sample[0:-1: steps, :]
    label = data[i + lookback + delay, 1]
    x_test.append(sample)
    y_test.append(label)

x_test = np.array(x_test)
y_test = np.array(y_test)
print(x_test.shape)
print(y_test.shape)

# model define
from keras.models import Model
from keras.layers import LSTM, Bidirectional, Dense, Flatten, Input, GRU, RNN

input = Input(shape=(120, 14))
x = Bidirectional(GRU(32, input_shape=(120, 14), dropout=0.1, recurrent_dropout=0.5, return_sequences=True))(input)
x = Bidirectional(GRU(32, input_shape=(120, 32), dropout=0.1, recurrent_dropout=0.5, activation='relu'))(x)
x = Dense(100)(x)
x = Dense(1)(x)

model = Model(inputs=[input], outputs=[x])
model.compile(optimizer='rmsprop', loss='mae')

# train
# model.fit([x_train], [y_train], epochs=10)
# model.save_weights("forecast.h5")
model.load_weights("forecast.h5")
# evaluate
y_result = model.predict_on_batch([x_train])

plt.plot(range(1000), y_train, color='r')
plt.plot(range(1000), y_result, color='b')
plt.show()
# display result
