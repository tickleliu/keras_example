import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.utils import plot_model

K.set_learning_phase(0)


def loss(y_true, y_pred):
    return K.sum(y_pred)

from keras.layers import Lambda

tf_print = Lambda(lambda x: tf.Print(x, [x.name, x]))
a = Input(shape=(1,))
b = Dense(2, activation="tanh")(a)
b = tf_print(b)
b = Dropout(rate=0.5)(b)
b = tf_print(b)
b = Dense(1, activation="tanh")(b)
b = tf_print(b)
b = BatchNormalization()(b)
b = tf_print(b)
m = Model(inputs=a, outputs=b)

m.compile(loss=loss, optimizer="SGD")
plot_model(model=m, to_file="m.png")

m.set_weights(
    [np.asarray([[1, 1]], dtype=np.float32), np.asarray([0., 0.], dtype=np.float32),
     np.asarray([[1],
                 [1]], dtype=np.float32), np.asarray([0.], dtype=np.float32),
     np.asarray([1.], dtype=np.float32),
     np.asarray([0.], dtype=np.float32),
     np.asarray([1.], dtype=np.float32),
     np.asarray([1.], dtype=np.float32)])
print(K.epsilon())

# m.set_weights(
#     [np.asarray([[1, 1]], dtype=np.float32), np.asarray([0., 0.], dtype=np.float32),#np.asarray([1., 1.], dtype=np.float32),
#      # np.asarray([0., 0.], dtype=np.float32), np.asarray([0., 0.], dtype=np.float32), np.asarray([1., 1.], dtype=np.float32),
#      np.asarray([[1],
#                  [1]], dtype=np.float32), np.asarray([0.], dtype=np.float32)])
print(m.get_weights())
# print(m.summary())

a_in = np.asarray([10])
print(m.predict(a_in))

# g = K.gradients(b, a)
# func = K.function(inputs=[a], outputs=[b])
# func1 = K.function(inputs=[a], outputs=[g[0]])
# print(func([np.asarray([[1]])]))
# print(func1([np.asarray([[1]])]))

# from numpy.random import rand
# from numpy import random
#
# random.seed(5)
# r = rand(2, 2, 2)
# print(r)
# r = K.variable(r, dtype=np.float32)
# result = K.sum(r, axis=-1)
# result = result.eval(session=K.get_session())
#
# result2 = K.sum(r, axis=2)
# result2 = K.eval(result2)
# print(result)
# print(result2)

# input1 = K.variable(value=0.1, dtype=np.float32)
# # input2 = K.variable(value=0.1, dtype=np.float32)
# out = input1 * input1 * 3 + input1 * 4
# g = K.gradients(out, input1)
# f = K.function(inputs=[input1], outputs=[g[0]])
#
# print(f([0.1]))
