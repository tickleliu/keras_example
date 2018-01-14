from keras.models import Sequential
from keras.layers import SimpleRNN, Embedding, Dense, LSTM, GRU, Bidirectional

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=32))
# model.add(SimpleRNN(32))
# model.add(GRU(32))
model.add(Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2)))
model.add(Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

from keras.datasets import imdb
from keras.preprocessing import sequence

num_words = 10000
maxlen = 20
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
x_train = sequence.pad_sequences(x_train, maxlen)
x_test = sequence.pad_sequences(x_train, maxlen)

model.fit(x_train, y_train, epochs=10, batch_size=100, validation_split=0.2)

model.save_weights("imdb.h5")
model.load_weights("imdb.h5")

acc = model.evaluate(x_test, y_test)
print(acc)
