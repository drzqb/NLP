'''
    IMDB fully based on Transformer
'''
import tensorflow as tf
import sys
import numpy as np
import matplotlib.pylab as plt
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
CLASSIFY = 2

tf.flags.DEFINE_integer('maxword', 100, 'max length of any sentences')
tf.flags.DEFINE_integer('block', 2, 'number of Encoder submodel')
tf.flags.DEFINE_integer('head', 8, 'number of multi_head attention')

tf.flags.DEFINE_integer('embedding_size', 200, 'size of dense representation of word')
tf.flags.DEFINE_float('keep_prob', 0.5, 'keep probility for rnn outputs')
tf.flags.DEFINE_integer('epochs', 20, 'number of iteration')
tf.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.flags.DEFINE_float('lr', 0.001, 'learning rate')
tf.flags.DEFINE_string('data', 'imdb.npz', 'directory of imdb data file')
FLAGS = tf.flags.FLAGS


def padding(x, padding_id):
    l = [len(xi) for xi in x]
    l_max = np.max(l)
    return [x[i] + [padding_id] * (l_max - l[i]) for i in range(len(x))]


def rearrange(x, r):
    return [x[ri] for ri in r]


def layer_norm_compute(x, scale, bias, epsilon=1.0e-8):
    """Layer norm raw computation."""
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias


def softmax(A, Mask):
    '''
    :param A: B*ML1*ML2
    :param Mask: B*ML1*ML2
    :return: C
    '''
    C = tf.exp(A)
    Cf = tf.zeros_like(C)
    C = tf.where(Mask, C, Cf)
    Cs = tf.reduce_sum(C, axis=-1, keepdims=True)
    C = tf.div(C, Cs)
    return C


def load_data():
    with np.load(FLAGS.data) as f:
        x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']

    print(x_train.shape, x_test.shape)

    m_train_samples = len(x_train)
    m_test_samples = len(x_test)

    for i in range(m_train_samples):
        x_train[i] = x_train[i][:(np.minimum(len(x_train[i]), FLAGS.maxword))]
    for i in range(m_test_samples):
        x_test[i] = x_test[i][:(np.minimum(len(x_test[i]), FLAGS.maxword))]

    x_train_length = [len(x_train[i]) for i in range(m_train_samples)]
    x_test_length = [len(x_test[i]) for i in range(m_test_samples)]
    vocab_size = np.max([np.max([np.max(x_train[i]) for i in range(m_train_samples)]),
                         np.max([np.max(x_test[i]) for i in range(m_test_samples)])]) + 1
    print(vocab_size, np.max(x_train_length), np.min(x_train_length), np.max(x_test_length), np.min(x_test_length))

    r = np.random.permutation(m_train_samples)
    x_train = rearrange(x_train, r)
    y_train = rearrange(y_train, r)

    r = np.random.permutation(m_test_samples)
    x_test = rearrange(x_test, r)
    y_test = rearrange(y_test, r)
    return vocab_size, x_train, y_train, x_test, y_test


def build_model(vocab_size):
    with tf.name_scope('Input'):
        x = tf.placeholder(tf.int32, [None, None], name='text')
        s = tf.placeholder(tf.int32, [None], name='sentiment')
        pos = tf.placeholder(tf.float32, [FLAGS.maxword, FLAGS.embedding_size], name='position')
        keep_prob = tf.placeholder(tf.float32, name='keep_probility')

        length = tf.reduce_sum(tf.sign(x), axis=-1, name='length')
        max_length = tf.reduce_max(length, name='max_length')

        padding_mask = tf.tile(tf.expand_dims(tf.greater(x, 0), 1), [FLAGS.head, tf.shape(x)[1], 1])
        sequence_mask = tf.tile(tf.expand_dims(tf.sequence_mask(length, max_length), 2), [1, 1, CLASSIFY])

    with tf.name_scope('Embedding'):
        embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, FLAGS.embedding_size], -1.0, 1.0),
                                       dtype=tf.float32,
                                       name='Embedding_Matrix')
        x_embedded = tf.nn.embedding_lookup(embedding_matrix, x)
        now = tf.nn.dropout(x_embedded + pos[:max_length], keep_prob)

    for block in range(FLAGS.block):
        with tf.variable_scope('selfattention' + str(block)):
            WQ = tf.layers.Dense(FLAGS.embedding_size, use_bias=False, name='Q')
            WK = tf.layers.Dense(FLAGS.embedding_size, use_bias=False, name='K')
            WV = tf.layers.Dense(FLAGS.embedding_size, use_bias=False, name='V')
            WO = tf.layers.Dense(FLAGS.embedding_size, use_bias=False, name='O')

            Q = tf.concat(tf.split(WQ(now), FLAGS.head, axis=-1), axis=0)
            K = tf.concat(tf.split(WK(now), FLAGS.head, axis=-1), axis=0)
            V = tf.concat(tf.split(WV(now), FLAGS.head, axis=-1), axis=0)

            QK = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(FLAGS.embedding_size / FLAGS.head)
            Z = tf.nn.dropout(WO(tf.concat(
                tf.split(tf.matmul(softmax(QK, padding_mask), V), FLAGS.head,
                         axis=0), axis=-1)), keep_prob)
            scale_sa = tf.get_variable('scale_sa',
                                       initializer=tf.ones([FLAGS.embedding_size], dtype=tf.float32))
            bias_sa = tf.get_variable('bias_sa',
                                      initializer=tf.zeros([FLAGS.embedding_size], dtype=tf.float32))
            now = layer_norm_compute(now + Z, scale_sa, bias_sa)
        with tf.variable_scope('feedforward' + str(block)):
            ffrelu = tf.layers.Dense(4 * FLAGS.embedding_size, activation=tf.nn.relu, name='ffrelu')
            ff = tf.layers.Dense(FLAGS.embedding_size, name='ff')
            scale_ff = tf.get_variable('scale_ff',
                                       initializer=tf.ones([FLAGS.embedding_size], dtype=tf.float32))
            bias_ff = tf.get_variable('bias_ff',
                                      initializer=tf.zeros([FLAGS.embedding_size], dtype=tf.float32))
            now = layer_norm_compute(ff(ffrelu(now)) + now, scale_ff, bias_ff)

    with tf.name_scope('Project'):
        output = tf.layers.dense(now, CLASSIFY)
        outputms = tf.zeros_like(output)

        logits = tf.reduce_sum(tf.where(sequence_mask, output, outputms), 1)
        logits = tf.div(logits, tf.cast(tf.expand_dims(length, axis=1), tf.float32), name='logits')

    with tf.name_scope('Loss'):
        prediction = tf.argmax(logits, axis=-1, output_type=tf.int32, name='prediction')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, s), tf.float32), name='accuracy')

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=s, logits=logits), name='loss')

        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr, name='optimizer')
        train_op = optimizer.minimize(loss, name='train_op')

    writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph(), filename_suffix='imdb4')
    writer.flush()
    writer.close()
    print('Graph saved successfully!')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    number_trainable_variables = 0
    variable_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variable_names)
    for k, v in zip(variable_names, values):
        print("Variable: ", k)
        print("Shape: ", v.shape)
        number_trainable_variables += np.prod([s for s in v.shape])
    print('Number of parameters: %d' % number_trainable_variables)

    return sess, x, s, pos, keep_prob, loss, train_op, prediction, accuracy


def train():
    vocab_size, x_train, y_train, x_test, y_test = load_data()
    sess, x, s, pos, keep_prob, loss, train_op, predict, accuracy = build_model(vocab_size)

    m_train_samples = len(x_train)
    total_batch = m_train_samples // FLAGS.batch_size

    x_test_batch = padding(x_test[:1000], 0)
    y_test_batch = y_test[:1000]

    loss_ = []
    acc_ = []
    pos_enc = np.array(
        [[position / np.power(10000.0, 2.0 * (i // 2) / FLAGS.embedding_size) for i in range(FLAGS.embedding_size)] for
         position in range(FLAGS.maxword)])
    pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
    pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])

    for epoch in range(1, FLAGS.epochs + 1):
        loss_epoch = 0.0
        acc_epoch = 0.0
        for batch in range(total_batch):
            x_batch = padding(x_train[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size], 0)
            y_batch = y_train[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size]
            feed_dict = {x: x_batch, s: y_batch, pos: pos_enc, keep_prob: FLAGS.keep_prob}

            loss_batch, predict_, acc_batch, _ = sess.run([loss, predict, accuracy, train_op],
                                                          feed_dict=feed_dict)
            loss_epoch += loss_batch
            acc_epoch += acc_batch
            sys.stdout.write('\r>> %d/%d | %d/%d | loss_batch: %.9f acc_batch: %.2f%%' % (
                epoch, FLAGS.epochs, batch + 1, total_batch, loss_batch, 100.0 * acc_batch))
            sys.stdout.flush()

        loss_.append(loss_epoch / total_batch)
        acc_.append(acc_epoch / total_batch)

        feed_dict = {x: x_test_batch, s: y_test_batch, pos: pos_enc, keep_prob: 1.0}
        acc = sess.run(accuracy, feed_dict)
        sys.stdout.write('    train_loss: %.9f train_acc: %.2f%%  test_acc: %.2f%%\n' % (
            loss_[-1], 100.0 * acc_[-1], 100.0 * acc))
        sys.stdout.flush()

        r = np.random.permutation(m_train_samples)
        x_train = rearrange(x_train, r)
        y_train = rearrange(y_train, r)

    plt.plot(loss_)
    plt.show()


def main(unused_argv):
    train()


if __name__ == '__main__':
    tf.app.run()
