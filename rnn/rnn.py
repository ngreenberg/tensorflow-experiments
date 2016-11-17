from __future__ import division

import numpy as np
import tensorflow as tf


#################
# Generate data #
#################

sequence_length = 10
num_features = 3
num_classes = 3
data_size = 10000

# generate inputs
input_data = np.random.choice(num_features, (data_size, sequence_length))


def classify(sequence):
    value = (sequence == 0).sum() + (sequence == 2).sum() * 4

    if value > 20:
        return 1
    elif value > 15:
        return 0
    else:
        return 2

# classify generated inputs
output_data = np.array([[classify(sequence[:i + 1])
                         for i in range(len(sequence))]
                        for sequence in input_data])

# convert data to one-hot encoding
input_data = np.eye(num_features)[input_data]
output_data = np.eye(num_classes)[output_data]

# split data into training data and testing data
train_fraction = .5
train_size = int(data_size * train_fraction)

train_input = input_data[:train_size]
train_output = output_data[:train_size]

test_input = input_data[train_size:]
test_output = output_data[train_size:]


###############
# Build model #
###############

data = tf.placeholder(tf.float32, [None, sequence_length, num_features])
target = tf.placeholder(tf.float32, [None, sequence_length, num_classes])

state_size = 50
cell = tf.nn.rnn_cell.LSTMCell(state_size)

output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

weight = tf.Variable(tf.random_normal([state_size, num_classes]))
bias = tf.Variable(tf.random_normal([num_classes]))

output = tf.reshape(output, [-1, state_size])
prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
prediction = tf.reshape(prediction, [-1, sequence_length, num_classes])
cross_entropy = -tf.reduce_sum(target * tf.log(prediction), [1, 2])
cross_entropy = tf.reduce_mean(cross_entropy)

learning_rate = 0.1
train_step = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(cross_entropy)


###############
# Train model #
###############

sess = tf.Session()
sess.run(tf.initialize_all_variables())

batch_size = 20
no_of_batches = len(train_input) // batch_size
epochs = 50
for epoch in range(epochs):
    if epoch % 5 == 0:
        print sess.run(cross_entropy,
                       feed_dict={data: test_input, target: test_output})

    ptr = 0
    for j in range(no_of_batches):
        inp = train_input[ptr:ptr + batch_size]
        out = train_output[ptr:ptr + batch_size]
        ptr += batch_size

        sess.run(train_step, feed_dict={data: inp, target: out})


##############
# Test model #
##############

output_classes = test_output.argmax(axis=2)
prediction_classes = sess.run(prediction,
                              feed_dict={data: test_input}).argmax(axis=2)
correct = (np.reshape(output_classes, -1) ==
           np.reshape(prediction_classes, -1)).sum()

print
print correct / len(np.reshape(output_classes, -1))
