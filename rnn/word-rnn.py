import tensorflow as tf

sequence_length = 4

raw_data = ['this is a test', 'I am a person', 'you are not here',
            'can I come down', 'I need to eat', 'you are very hungry']

unique_words = set(' '.join(raw_data).split(' '))
vocab = {word: i + 1 for i, word in enumerate(unique_words)}

indexed_data = [[vocab[word] for word in line.split(' ')] for line in raw_data]

input_x = tf.placeholder(tf.int32, [None, sequence_length])

vocab_size = len(vocab) + 1
embedding_size = 10

embedding_matrix = tf.Variable(
    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
embedded_data = tf.nn.embedding_lookup(embedding_matrix, input_x)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
print sess.run(embedded_data, feed_dict={input_x: indexed_data})
