import tensorflow as tf


def random_variables(shape):
    return tf.Variable(tf.random_normal(shape))


def build_neural_net(input_size, output_size, hidden_sizes):
    weight_shapes = zip([input_size] + hidden_sizes,
                        hidden_sizes + [output_size])
    weights = [random_variables(shape) for shape in weight_shapes]

    biases = [random_variables([size])
              for size in hidden_sizes + [output_size]]

    return weights, biases


def cross_entropy(y, y_hat):
    return tf.reduce_mean(
        -tf.reduce_sum(y * tf.log(y_hat), reduction_indices=[1]))


class SigmoidNeuralNet:

    def __init__(self, input_size, output_size, hidden_sizes):
        self.x = tf.placeholder(tf.float32, [None, input_size])
        self.y = tf.placeholder(tf.float32, [None, output_size])

        self.weights, self.biases = build_neural_net(
            input_size, output_size, hidden_sizes)

    def run(self, input_data):
        output = input_data
        for i in range(len(self.weights) - 1):
            output = tf.nn.sigmoid(
                tf.matmul(output, self.weights[i]) + self.biases[i])
        output = tf.nn.softmax(
            tf.matmul(output, self.weights[-1]) + self.biases[-1])

        return output

    def predict(self, input_data):
        y_hat = self.run(self.x)
        predictions = tf.argmax(y_hat, dimension=1)

        return self.sess.run(predictions, feed_dict={self.x: input_data})

    def train(self, input_data, output_data,
              learning_rate, training_epochs, batch_size):
        y_hat = self.run(self.x)
        cost = cross_entropy(self.y, y_hat)
        train_step = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(cost)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

        for i in range(10):
            print str(i * 10) + "% trained - cost:",
            print self.sess.run(cost, feed_dict={
                                self.x: input_data, self.y: output_data})

            for epoch in range(training_epochs / 10):
                self.sess.run(train_step, feed_dict={
                              self.x: input_data, self.y: output_data})

        print "100% trained - cost:",
        print self.sess.run(cost, feed_dict={
                            self.x: input_data, self.y: output_data})
        print

if __name__ == '__main__':
    XOR_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    XOR_y = [[1, 0], [0, 1], [0, 1], [1, 0]]

    nn = SigmoidNeuralNet(2, 2, [5, 10, 20, 10, 5])
    nn.train(XOR_x, XOR_y, .01, 100000, 0)

    print nn.predict(XOR_x)
