import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(os.path.join(os.getcwd(), "MNIST/"), one_hot=True)

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
    from keras.layers import Flatten

    # x = tf.Print(x, [tf.shape(x)])
    x_ = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv_1 = Conv2D(32, (5, 5), strides=(1, 1), use_bias=True, kernel_regularizer=regularizers.l2(0.001))(x_)
    conv_1 = MaxPool2D()(conv_1)
    conv_2 = Conv2D(32, (3, 3), strides=(1, 1), use_bias=True, kernel_regularizer=regularizers.l2(0.001))(conv_1)
    conv_2 = MaxPool2D()(conv_2)
    conv_2 = Flatten()(conv_2)
    dense_3 = Dense(32, activation="relu")(conv_2)
    dense_4 = Dense(10, activation="softmax")(dense_3)
    dense_4 = tf.Print(dense_4, [tf.shape(dense_4)])

    # # keras model
    # from keras.models import Model
    # model = Model(inputs=[x], outputs=[dense_4])
    #
    # from keras.optimizers import RMSprop
    # from keras.optimizers import Adadelta
    # from keras.optimizers import Adam
    # from keras.optimizers import Adagrad
    # from keras.optimizers import SGD
    # model.compile(optimizer=SGD(lr=0.01, decay=0.001, momentum=0.5), loss='category_crossentropy', metrics=['accuracy'])
    # train_img, train_label = mnist.train.next_batch(100)
    # model.fit(x=[train_img], y=[train_label],validation_split=0.1)

    from keras.objectives import categorical_crossentropy

    loss = tf.reduce_mean(categorical_crossentropy(y, dense_4))

    train_step = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.5).minimize(loss)
    equal_result = tf.equal(tf.argmax(y, 1), tf.argmax(dense_4, 1))
    result = tf.cast(equal_result, dtype=tf.float32)
    accuracy = tf.reduce_mean(result)

    writer = tf.summary.FileWriter('./keras_tensorflow_log/')
    outloss = tf.summary.scalar('loss', loss)
    merge = tf.summary.merge([outloss])
    saver = tf.train.Saver()

    init = [tf.local_variables_initializer(), tf.global_variables_initializer()]
    sess.run(init)
    for i in range(1000):
        train_img, train_label = mnist.train.next_batch(100)
        ls, summ = sess.run([train_step, merge], feed_dict={x: train_img, y: train_label})
        writer.add_summary(summary=summ, global_step=i)
        if i % 10 == 0:
            test_img, test_label = mnist.test.next_batch(100)
            acc = sess.run(accuracy, feed_dict={x: test_img, y: test_label})
            print("accuracy: %f" % acc)


    saver_path = saver.save(sess, "./keras_tensorflow_log/model.ckpt")
