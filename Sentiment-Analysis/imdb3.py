'''
    IMDB based on Transformer
    The result of experiment shows the Transformer model doesn't perform well for very long text instance.
'''
import tensorflow as tf
import sys
import numpy as np
import matplotlib.pylab as plt
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.flags.DEFINE_integer('maxword', 20, 'max length of any sentences')
tf.flags.DEFINE_integer('embedding_size', 512, 'size of dense representation of word')
tf.flags.DEFINE_float('keep_prob', 1.0, 'keep probility for rnn outputs')
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
        padding_mask = tf.tile(tf.expand_dims(tf.greater(x, 0), 1), [1, tf.shape(x)[1], 1], name='mask')

    with tf.name_scope('Embedding'):
        embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, FLAGS.embedding_size], -1.0, 1.0),
                                       dtype=tf.float32,
                                       name='Embedding_Matrix')
        x_embedded = tf.nn.embedding_lookup(embedding_matrix, x)
        now = tf.nn.dropout(x_embedded + pos[:max_length], keep_prob)

    for block in range(1):
        with tf.name_scope('SelfAttention' + str(block)):
            WQ = [tf.layers.Dense(FLAGS.embedding_size / 8, use_bias=False, name='Q' + str(block) + str(i)) for i in
                  range(8)]
            WK = [tf.layers.Dense(FLAGS.embedding_size / 8, use_bias=False, name='K' + str(block) + str(i)) for i in
                  range(8)]
            WV = [tf.layers.Dense(FLAGS.embedding_size / 8, use_bias=False, name='V' + str(block) + str(i)) for i in
                  range(8)]
            WO = tf.layers.Dense(FLAGS.embedding_size, use_bias=False, name='O' + str(block))

            Q = [WQ[i](now) for i in range(8)]
            K = [WK[i](now) for i in range(8)]
            V = [WV[i](now) for i in range(8)]

            QK = [tf.matmul(Q[i], tf.transpose(K[i], [0, 2, 1])) / tf.sqrt(FLAGS.embedding_size / 8) for i in range(8)]
            QKF = [tf.ones_like(QK[i]) ** (1. - tf.pow(2., 31.)) for i in range(8)]
            Z = tf.nn.dropout(WO(tf.concat(
                [tf.matmul(tf.nn.softmax(tf.where(padding_mask, QK[i], QKF[i]), axis=-1), V[i]) for i in range(8)],
                axis=-1)), keep_prob)

        with tf.name_scope('ResNorm' + str(block)):
            scale_rn = tf.Variable(tf.ones([FLAGS.embedding_size], dtype=tf.float32), name='scale_rn')
            bias_rn = tf.Variable(tf.zeros([FLAGS.embedding_size], dtype=tf.float32), name='bias_rn')

            selfatten_output = tf.nn.dropout(layer_norm_compute(now + Z, scale_rn, bias_rn), keep_prob)

        with tf.name_scope('FeedForward' + str(block)):
            ffrelu = tf.layers.Dense(FLAGS.embedding_size, activation=tf.nn.relu, name='ffrelu' + str(block))
            ff = tf.layers.Dense(FLAGS.embedding_size, name='ff' + str(block))
            scale_ff = tf.Variable(tf.ones([FLAGS.embedding_size], dtype=tf.float32), name='scale_ff')
            bias_ff = tf.Variable(tf.zeros([FLAGS.embedding_size], dtype=tf.float32), name='bias_ff')

            now = tf.nn.dropout(
                layer_norm_compute(ff(ffrelu(selfatten_output)) + selfatten_output, scale_ff, bias_ff), keep_prob)

    with tf.name_scope('Project'):
        project = tf.layers.Dense(2, name='project')
        sequence_mask = tf.tile(tf.expand_dims(tf.sequence_mask(length, max_length), 2), [1, 1, 2])
        output = project(now)
        outputms = tf.zeros_like(output)

        logits = tf.reduce_sum(tf.where(sequence_mask, output, outputms), 1)
        logits = tf.transpose(tf.div(tf.transpose(logits), tf.cast(length, tf.float32)), name='prediction')

    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=s, logits=logits), name='loss')
        train_op = tf.train.RMSPropOptimizer(learning_rate=FLAGS.lr).minimize(loss)

        predict = tf.argmax(logits, 1, name='predict', output_type=tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, s), tf.float32))

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
    sess.close()

    return x, s, pos, keep_prob, loss, train_op, predict, accuracy


def train():
    vocab_size, x_train, y_train, x_test, y_test = load_data()
    x, s, pos, keep_prob, loss, train_op, predict, accuracy = build_model(vocab_size)

    m_train_samples = len(x_train)
    total_batch = m_train_samples // FLAGS.batch_size

    x_test_batch = padding(x_test[:1000], 0)
    y_test_batch = y_test[:1000]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

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

            tmp_loss_epoch, predict_, tmp_acc_epoch, _ = sess.run([loss, predict, accuracy, train_op],
                                                                  feed_dict=feed_dict)
            loss_epoch += tmp_loss_epoch
            acc_epoch += tmp_acc_epoch
            sys.stdout.write('\r>> %d/%d | %d/%d | loss: %.9f acc: %.2f%%' % (
                epoch, FLAGS.epochs, batch + 1, total_batch, tmp_loss_epoch, 100.0 * tmp_acc_epoch))
            sys.stdout.flush()

        loss_.append(loss_epoch / total_batch)
        acc_.append(acc_epoch / total_batch)

        feed_dict = {x: x_test_batch, s: y_test_batch, pos: pos_enc, keep_prob: 1.0}
        acc = sess.run(accuracy, feed_dict)
        sys.stdout.write('    train_loss:%.9f train_accuracy:%.2f%%  test_accuracy:%.2f%%\n' % (
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
