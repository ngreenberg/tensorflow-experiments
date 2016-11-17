from __future__ import division

import numpy as np
from numpy.random import choice

import tensorflow as tf


#################
# Generate data #
#################

sequence_length = 15

numbers = np.array([choice(xrange(1, 6),
                           choice(xrange(8, sequence_length + 1)))
                    for _ in xrange(2000)])

lengths = [len(sequence) for sequence in numbers]


def classify_sequence(sequence):
    last_index = len(sequence) - 1

    labels = [sequence[i - 1] + sequence[i] + sequence[i + 1]
              for i in xrange(1, last_index)]

    first_label = 5 + sequence[0] + sequence[1]
    last_label = sequence[last_index - 1] + sequence[last_index] + 5
    labels = [first_label] + labels + [last_label]

    labels = [n // 3 for n in labels]

    labels = np.eye(6)[labels]
    labels = labels[:, 1:]

    padded = np.zeros((sequence_length, 5), dtype=np.int)
    padded[:len(labels)] = labels

    return padded


output_data = np.array([classify_sequence(sequence) for sequence in numbers])

words = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five'}

raw_data = [' '.join([words[n] for n in sequence]) for sequence in numbers]


################
# Process data #
################

unique_words = set(' '.join(raw_data).split(' '))
vocab = {word: i + 1 for i, word in enumerate(unique_words)}

indexed_data = [[vocab[word] for word in line.split(' ')] for line in raw_data]


def pad(sequence):
    padded = np.zeros(sequence_length, dtype=np.int)
    padded[:len(sequence)] = sequence
    return padded


input_data = np.array([pad(sequence) for sequence in indexed_data])

train_input = input_data[:1000]
train_output = output_data[:1000]

test_input = input_data[1000:]
test_output = output_data[1000:]

train_lengths = lengths[:1000]
test_lengths = lengths[1000:]


###############
# Build model #
###############

num_classes = 5

input_x = tf.placeholder(tf.int32, [None, sequence_length])


def length(sequence):
    used = tf.sign(tf.abs(sequence))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


batch_length = length(input_x)

vocab_size = len(vocab) + 1
embedding_size = 10

embedding_matrix = tf.Variable(tf.random_uniform(
    [vocab_size, embedding_size], -1.0, 1.0))
embedded_data = tf.nn.embedding_lookup(embedding_matrix, input_x)

target = tf.placeholder(tf.float32, [None, sequence_length, num_classes])

state_size = 10
fw_cell = tf.nn.rnn_cell.LSTMCell(state_size)
bw_cell = tf.nn.rnn_cell.LSTMCell(state_size)

outputs, state = tf.nn.bidirectional_dynamic_rnn(
    fw_cell, bw_cell, embedded_data,
    sequence_length=batch_length, dtype=tf.float32)

fw_output, bw_output = outputs
output = tf.concat(2, [fw_output, bw_output])

weight = tf.Variable(tf.random_normal([state_size * 2, num_classes]))
bias = tf.Variable(tf.random_normal([num_classes]))

output = tf.reshape(output, [-1, state_size * 2])
prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
prediction = tf.reshape(prediction, [-1, sequence_length, num_classes])


def cost(output, target):
    # Compute cross entropy for each frame.
    cross_entropy = target * tf.log(output)
    cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
    mask = tf.sign(tf.reduce_max(tf.abs(target), reduction_indices=2))
    cross_entropy *= mask
    # Average over actual sequence lengths.
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
    cross_entropy /= tf.reduce_sum(mask, reduction_indices=1)
    return tf.reduce_mean(cross_entropy)


cross_entropy = cost(prediction, target)
# cross_entropy = -tf.reduce_sum(target * tf.log(prediction), [1, 2])
# cross_entropy = tf.reduce_mean(cross_entropy)

learning_rate = 0.1
train_step = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(cross_entropy)


###############
# Train model #
###############

sess = tf.Session()
sess.run(tf.initialize_all_variables())

batch_size = 20
no_of_batches = len(input_data) // batch_size
epochs = 100
for epoch in range(epochs):
    if epoch % 10 == 0:
        print sess.run(cross_entropy,
                       feed_dict={input_x: train_input, target: train_output})

    ptr = 0
    for j in range(no_of_batches):
        inp = train_input[ptr:ptr + batch_size]
        out = train_output[ptr:ptr + batch_size]
        ptr += batch_size

        sess.run(train_step, feed_dict={input_x: inp, target: out})


##############
# Test model #
##############

output_classes = test_output.argmax(axis=2)
prediction_classes = sess.run(prediction,
                              feed_dict={input_x: test_input}).argmax(axis=2)

output_classes = [classes[:test_lengths[i]]
                  for i, classes in enumerate(output_classes)]
output_classes_all = np.concatenate(output_classes)

prediction_classes = [classes[:test_lengths[i]]
                      for i, classes in enumerate(prediction_classes)]
prediction_classes_all = np.concatenate(prediction_classes)

correct = (output_classes_all == prediction_classes_all).sum()

print
print correct / len(output_classes_all)
