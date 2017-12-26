import tensorflow as tf
from tensorflow.python import debug as tf_debug

q = tf.FIFOQueue(3, "float32")
# init =q.enqueue(tf.constant([0.1, 0.2, 0.3]))
init =q.enqueue_many(([0.1, 0.2, 0.3], ))

x = q.dequeue()
y = x + 1
q_inc = q.enqueue([y])
t = tf.constant([[1.0, 0, 0, 0], [0, 1, 0, 0]])
# t = tf.expand_dims(t, 0)
t_1 = tf.constant([[0, 0, 1.0, 0], [0, 1, 0, 0]])
# t_1 = tf.expand_dims(t_1, 0)

with tf.Session() as sess:
    arg = tf.argmax(t,  dimension=1)
    arg_1 = tf.argmax(t_1, dimension=1)
    equal_result = tf.equal(arg, arg_1)
    result = tf.cast(equal_result, dtype=tf.float32)
    mean = tf.reduce_mean(result)
    # print(sess.run([t, t_1, tf.shape(t), arg, arg_1, equal_result, result, mean]))
    print(sess.run([arg, arg_1, equal_result, result, mean]))

#     sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# # with tf_debug.LocalCLIDebugWrapperSession(tf.Session()) as sess:
#     sess.run(init)
#     quelen = sess.run(q.size())
#     for i in range(2):
#         sess.run(q_inc)
#     quelen = sess.run(q.size())
#     for i in range(quelen):
#         print(sess.run(q.dequeue()))

