import tensorflow as tf

import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder('float', [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, w) + b)
y_ = tf.placeholder('float', [None, 10])

cross_entropy = - tf.reduce_sum(y_ * tf.log(y))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)
for step in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size=100)
    sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
sess.close()
