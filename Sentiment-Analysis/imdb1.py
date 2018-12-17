import tensorflow as tf
import sys
import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.flags.DEFINE_integer('maxword', 400, 'max length of any sentences')
tf.flags.DEFINE_multi_integer('rnn_size', [128, 64], 'size of GRU cells of layers')
tf.flags.DEFINE_integer('embedding_size', 200, 'size of dense representation of word')
tf.flags.DEFINE_integer('epochs', 20, 'number of iteration')
tf.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.flags.DEFINE_float('lr', 0.001, 'learning rate')
tf.flags.DEFINE_string('data', 'imdb.npz', 'directory of imdb data file')
FLAGS = tf.flags.FLAGS


def create_rnn_cell(rnn_size, layers, keep_prob):
    def single_rnn_cell(rnn_size, keep_prob):
        single_cell = tf.nn.rnn_cell.GRUCell(rnn_size)
        cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, output_keep_prob=keep_prob)
        return cell

    cell = tf.nn.rnn_cell.MultiRNNCell([single_rnn_cell(rnn_size[i], keep_prob) for i in range(layers)])
    return cell


def padding(x, l, padding_id):
    l_max = np.max(l)
    return [x[i] + [padding_id] * (l_max - l[i]) for i in range(len(x))]


def rearrange(x, r):
    return [x[ri] for ri in r]


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(FLAGS.data)
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

with tf.name_scope('Input'):
    x_batch = tf.placeholder(tf.int32, [None, None], name='Text')
    y_batch = tf.placeholder(tf.int32, [None], name='Sentiment')
    x_length = tf.placeholder(tf.int32, [None], name='Text_Length')
    keep_prob = tf.placeholder(tf.float32, name='Keep_Probility')

with tf.name_scope('Embedding'):
    embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, FLAGS.embedding_size], -1.0, 1.0), dtype=tf.float32,
                                   name='Embedding_Matrix')
    x_batch_embedded = tf.nn.embedding_lookup(embedding_matrix, x_batch)

    gru_cell = create_rnn_cell(FLAGS.rnn_size, len(FLAGS.rnn_size), keep_prob)
    _, final_state = tf.nn.dynamic_rnn(gru_cell, inputs=x_batch_embedded, sequence_length=x_length, dtype=tf.float32)
    final_output = final_state[-1]
    output = tf.layers.Dense(2)(final_output)

with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_batch, logits=output), name='loss')
    train_op = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.argmax(output, 1, name='predict', output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y_batch), tf.float32))

total_batch = m_train_samples // FLAGS.batch_size

x_test_length_batch = x_test_length[:1000]
x_test_batch = padding(x_test[:1000], x_test_length_batch, 0)
y_test_batch = y_test[:1000]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

loss_ = []
for epoch in range(1, FLAGS.epochs + 1):
    loss_epoch = 0.0

    for batch in range(total_batch):
        sys.stdout.write('\r>> %d/%d | %d/%d' % (epoch, FLAGS.epochs, batch + 1, total_batch))
        sys.stdout.flush()
        x_train_length_batch = x_train_length[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size]
        x_train_batch = padding(x_train[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size], x_train_length_batch,
                                0)
        y_train_batch = y_train[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size]
        feed_dict = {x_batch: x_train_batch, y_batch: y_train_batch, x_length: x_train_length_batch, keep_prob: 0.8}
        tmp_loss_epoch, _ = sess.run([loss, train_op], feed_dict)
        loss_epoch += tmp_loss_epoch
    loss_.append(loss_epoch / total_batch)

    feed_dict = {x_batch: x_test_batch, y_batch: y_test_batch, x_length: x_test_length_batch, keep_prob: 1.0}
    acc = sess.run(accuracy, feed_dict)
    sys.stdout.write('  Loss:%f Accuracy:%.2f%%\n' % (loss_[-1], 100.0 * acc))
    sys.stdout.flush()

    r = np.random.permutation(m_train_samples)
    x_train = rearrange(x_train, r)
    x_train_length = rearrange(x_train_length, r)
    y_train = rearrange(y_train, r)

plt.plot(loss_)
plt.show()
