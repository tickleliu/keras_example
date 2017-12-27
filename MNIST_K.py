'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf

batch_size = 128
num_classes = 10
epochs = 2

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(type(x_train))
print(y_train.shape)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x = Input(shape=(28, 28, 1))
model = Sequential()
conv1 = Conv2D(32, kernel_size=(3, 3),
               activation='relu', )(x)
conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv2)
pool3 = Dropout(0.25)(pool3)
pool3 = Flatten()(pool3)
fc4 = Dense(128, activation='relu')(pool3)
fc4 = Dropout(0.5)(fc4)
print(type(fc4))
out = Dense(num_classes, activation='softmax')(fc4)

tb = keras.callbacks.TensorBoard(log_dir="keras_tensorflow_log", batch_size=batch_size)
model = keras.models.Model(inputs=[x], outputs=[out])
from keras import  metrics
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy', metrics.categorical_accuracy])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True,
          verbose=1,
          validation_data=(x_test, y_test), callbacks=[tb])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
