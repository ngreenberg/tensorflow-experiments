import tensorflow as tf
import numpy as np

input_size = 2
h1_size = 5
h2_size = 5
output_size = 2

learning_rate = 0.01
training_epochs = 100000

x = tf.placeholder(tf.float32, [None, input_size])
y = tf.placeholder(tf.float32, [None, output_size])


def random_variables(shape):
    return tf.Variable(tf.random_normal(shape))

weights = {
    'h1': random_variables([input_size, h1_size]),
    'h2': random_variables([h1_size, h2_size]),
    'out': random_variables([h2_size, output_size])
}
biases = {
    'h1': random_variables([h1_size]),
    'h2': random_variables([h2_size]),
    'out': random_variables([output_size])
}


def model(x, weights, biases):
    h1 = tf.nn.sigmoid(tf.matmul(x, weights['h1']) + biases['h1'])
    h2 = tf.nn.sigmoid(tf.matmul(h1, weights['h2']) + biases['h2'])
    out = tf.nn.softmax(tf.matmul(h2, weights['out']) + biases['out'])
    return out

y_hat = model(x, weights, biases)
predictions = tf.argmax(y_hat, dimension=1)

cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(y * tf.log(y_hat), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(cross_entropy)

init = tf.initialize_all_variables()

XOR_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_y = [[1, 0], [0, 1], [0, 1], [1, 0]]

with tf.Session() as sess:
    sess.run(init)

    for i in range(10):
        print str(i * 10) + "% trained - cost:",
        print sess.run(cross_entropy, feed_dict={x: XOR_x, y: XOR_y})

        for epoch in range(training_epochs / 10):
            sess.run(train_step, feed_dict={x: XOR_x, y: XOR_y})

    print "x:\t\t", XOR_x
    print "predicted y:\t", sess.run(predictions, feed_dict={x: XOR_x})
    print "y:\t\t", np.argmax(XOR_y, axis=1)
