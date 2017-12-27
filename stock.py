import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow

f = open('./dataset/dataset_1.csv')
# f = open('./dataset/dataset_2.csv')
df = pd.read_csv(f)  # 读入股票数据
data = np.array(df['最高价'])  # 获取最高价序列
data = data[::-1]  # 反转，使数据按照日期先后顺序排列
# 以折线图展示data
# plt.figure()
# plt.plot(data)
# plt.show()
normalize_data = (data - np.mean(data)) / np.std(data)  # 标准化
normalize_data = normalize_data[:, np.newaxis]  # 增加维度
# ———————————————————形成训练集—————————————————————
# 设置常量
time_step = 20  # 时间步
rnn_unit = 10  # hidden layer units
batch_size = 60  # 每一批次训练多少个样例
input_size = 1  # 输入层维度
output_size = 1  # 输出层维度
lr = 0.0006  # 学习率
train_x, train_y = [], []  # 训练集
for i in range(len(normalize_data) - time_step - 1):
    x = normalize_data[i:i + time_step]
    y = normalize_data[i + 1:i + time_step + 1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())
train_x = np.array(train_x)
train_y = np.array(train_y)
print(train_x.shape)

from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Dense
from keras.layers import TimeDistributed

# model =  Sequential()
# model.add(LSTM(batch_input_shape=(10, time_step, 1), output_dim=rnn_unit,return_sequences=True, stateful=True))
# model.add(TimeDistributed(Dense(1)))

x = Input((time_step, 1))
lstm = LSTM(input_shape=(time_step, 1), output_dim=10, return_sequences=True)(x)
lstm = LSTM(input_shape=(time_step, 10), output_dim=10, return_sequences=True)(lstm)
out = TimeDistributed(Dense(1))(lstm)
model = Model(inputs=[x], output=[out])

from keras.optimizers import Adam

from keras.metrics import mae
model.compile(loss="mse", optimizer=Adam(), metrics=[mae])
# model.fit(train_x, train_y, epochs=2, batch_size=10)

# model.save("./stock.h5")
model.load_weights("./stock.h5")
y = model.predict_on_batch(train_x)
print(y.shape)
result = []
for i in range(y.shape[0]):
    result.append(y[i, 19, 0])
result = np.array(result)
print(result.shape)
plt.figure()
plt.subplot(211)
plt.plot(data)
plt.subplot(212)
plt.plot(result)
plt.show()