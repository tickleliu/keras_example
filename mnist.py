import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/home/liuml/keras/MNIST/", one_hot=True)

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import tensorflow.contrib.slim as slim

with tf.Session() as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    # model
    from keras import backend as K

    K.set_session(sess)
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    from keras.layers import Dense
    from keras.layers import Conv2D
    from keras.layers import MaxPool2D
    from keras import regularizers
    x = tf.Print(x, [tf.shape(x)])
    x_ = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv_1 = Conv2D(32, (5, 5), strides=(1, 1), use_bias=True, kernel_regularizer=regularizers.l2(0.001))(x_)
    conv_1 = MaxPool2D()(conv_1)
    conv_2 = Conv2D(32, (3, 3), strides=(1, 1), use_bias=True, kernel_regularizer=regularizers.l2(0.001))(conv_1)
    conv_2 = MaxPool2D()(conv_2)
    dense_3 = Dense(32, activation="relu")(conv_2)
    dense_4 = Dense(10, activation="softmax")(dense_3)
    from keras.objectives import categorical_crossentropy

    loss = tf.reduce_mean(categorical_crossentropy(y, dense_4))

    train_step = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.5).minimize(loss)
    equal_result = tf.equal(tf.argmax(y, 1), tf.argmax(dense_4, 1))
    result = tf.cast(equal_result, dtype=tf.float32)
    accuracy = tf.reduce_mean(result)

    init = [tf.local_variables_initializer(), tf.global_variables_initializer()]
    sess.run(init)
    train_img, train_label = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: train_img, y: train_label})
