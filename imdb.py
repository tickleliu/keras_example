import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from keras.layers import Embedding, Flatten, Dense
from keras import preprocessing
from keras.datasets import imdb

max_feature = 10000
maxlen = 20

(x_train, y_train), (x_test, x_label) = imdb.load_data(num_words=max_feature)
print(x_train.shape)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
print(x_train.shape)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

from keras.models import Sequential
model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

# histroy = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
# model.save_weights("seq.h5")
model.load_weights("seq.h5")
embedding_layer = model.get_layer("embedding_1")
from keras import backend as K
# from keras.models import Model
# model_embedding = Model(inputs=model.input, outputs=[embedding_layer.output])
# print(model_embedding.predict(x_test[0: 1, :]))

embedding_result = K.function([model.input], [embedding_layer.output])
# print(x_test.shape)
# import numpy as np
#
print(embedding_result([x_test[0:1, :]]))
