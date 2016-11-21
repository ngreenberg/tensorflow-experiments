import functools

import tensorflow as tf

# Module structure based off of
# https://danijar.com/structuring-your-tensorflow-models/

# Variable sequence length algorithms based off of
# https://danijar.com/variable-sequence-lengths-in-tensorflow/


def lazy_property(function):

    attribute = '__cache__' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class BiLSTM(object):

    def __init__(self, data, target,
                 vocab_size, embedding_size=100, lstm_size=100):
        self.data = data
        self.target = target

        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._lstm_size = lstm_size

        self.prediction
        self.optimize
        self.error

    @lazy_property
    def length(self):
        used = tf.sign(tf.abs(self.data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def prediction(self):
        # Word embeddings
        embedding_matrix = self._random_variable(
            [self._vocab_size, self._embedding_size], uniform=True)
        embedded_data = tf.nn.embedding_lookup(embedding_matrix, self.data)

        # Bidirectional RNN
        output, _ = tf.nn.bidirectional_dynamic_rnn(
            tf.nn.rnn_cell.LSTMCell(self._lstm_size),
            tf.nn.rnn_cell.LSTMCell(self._lstm_size),
            embedded_data, sequence_length=self.length, dtype=tf.float32)

        fw_output, bw_output = output
        output = tf.concat(2, [fw_output, bw_output])

        # Softmax layer
        max_length = int(self.target.get_shape()[1])
        num_classes = int(self.target.get_shape()[2])

        weight = self._random_variable([self._lstm_size * 2, num_classes])
        bias = self._random_variable([num_classes])

        output = tf.reshape(output, [-1, self._lstm_size * 2])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        prediction = tf.reshape(prediction, [-1, max_length, num_classes])

        return prediction

    @lazy_property
    def cost(self):
        # Compute cross entropy for each frame
        cross_entropy = self.target * tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
        cross_entropy *= mask

        # Average over actual sequence lengths
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(cross_entropy)

    @lazy_property
    def optimize(self):
        learning_rate = 0.1
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.target, 2),
                                tf.argmax(self.prediction, 2))
        mistakes = tf.cast(mistakes, tf.float32)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
        mistakes *= mask

        # Average over actual sequence lengths.
        mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
        mistakes /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(mistakes)

    @staticmethod
    def _random_variable(shape, uniform=False):
        if uniform:
            return tf.Variable(tf.random_uniform(shape, -1.0, 1.0))
        else:
            return tf.Variable(tf.random_normal(shape))
