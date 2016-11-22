from __future__ import division

import numpy as np
from numpy.random import choice

import tensorflow as tf
from bilstm import BiLSTM


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
data = tf.placeholder(tf.int32, [None, sequence_length])
target = tf.placeholder(tf.float32, [None, sequence_length, num_classes])

model = BiLSTM(data, target, len(vocab) + 1, embedding_size=10, lstm_size=10)


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
        print sess.run(model.cost,
                       feed_dict={data: train_input, target: train_output})

    ptr = 0
    for j in range(no_of_batches):
        inp = train_input[ptr:ptr + batch_size]
        out = train_output[ptr:ptr + batch_size]
        ptr += batch_size

        sess.run(model.optimize, feed_dict={data: inp, target: out})


##############
# Test model #
##############

print
print sess.run(model.error, feed_dict={data: test_input, target: test_output})
