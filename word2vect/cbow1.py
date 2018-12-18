'''
    word2vec CBOW model based on tensorflow
'''

import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import os, sys, zipfile, collections
from sklearn.manifold import TSNE
from cbow_hparams import FLAGS

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Lang():
    def __init__(self):
        text = self.read_data()
        word2ind = self.create_dict(text)
        len_text, text2ind = self.text2id(text, word2ind)
        self.make_data(len_text, text2ind)

    def read_data(self):
        with zipfile.ZipFile(FLAGS.corpus) as f:
            data = f.read(f.namelist()[0]).decode('utf-8').split()
        return data

    def create_dict(self, text):
        counter = collections.Counter(text)
        most_vocab = counter.most_common(FLAGS.vocab_size)

        word2ind = dict()
        word2ind['UNK'] = 0
        for i in range(FLAGS.vocab_size - 1):
            word2ind[most_vocab[i][0]] = i + 1

        self.ind2word = {v: k for k, v in word2ind.items()}

        return word2ind

    def text2id(self, text, word2ind):
        len_text = len(text)
        text2ind = [word2ind[text[i]] if text[i] in word2ind.keys() else 0 for i in range(len_text)]
        return len_text, text2ind

    def make_data(self, len_text, text2ind):
        self.x_train = np.array([[text2ind[i + k] for k in range(FLAGS.window_size)] +
                                 [text2ind[i + FLAGS.window_size + 1 + k] for k in range(FLAGS.window_size)]
                                 for i in range(0, len_text - 2 * FLAGS.window_size)])
        self.y_train = np.array([text2ind[i] for i in range(FLAGS.window_size, len_text - FLAGS.window_size)]).reshape(
            (-1, 1))


class CBOW():
    def __init__(self):
        self.build()

    def build(self):
        with tf.name_scope('Input'):
            self.train_inputs = tf.placeholder(tf.int32, [None, 2 * FLAGS.window_size], name='context')
            self.train_labels = tf.placeholder(tf.int32, [None, 1], name='word')
        with tf.name_scope('Embed'):
            embed_matrix = tf.Variable(tf.random_uniform([FLAGS.vocab_size, FLAGS.embed_size], -1.0, 1.0),
                                       name='embed_matrix')
            embed = tf.nn.embedding_lookup(embed_matrix, self.train_inputs, name='embed')
        with tf.name_scope('Mean'):
            kmean = tf.reduce_mean(embed, axis=1, name='average')
        with tf.name_scope('LossTrain'):
            nce_weights = tf.Variable(
                tf.truncated_normal([FLAGS.vocab_size, FLAGS.embed_size], stddev=1.0 / np.sqrt(FLAGS.embed_size)),
                name='weights')
            nce_biases = tf.Variable(tf.zeros([FLAGS.vocab_size]), name='biase')
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=self.train_labels,
                                                      inputs=kmean, num_sampled=FLAGS.num_samples,
                                                      num_classes=FLAGS.vocab_size),
                                       name='loss')
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.lr).minimize(self.loss)

        with tf.name_scope('normalize'):
            norm = tf.sqrt(tf.reduce_sum(tf.square(embed_matrix), axis=1, keepdims=True), name='norm')
            self.norm_matrix = embed_matrix / norm

        writer = tf.summary.FileWriter(logdir='logs', graph=tf.get_default_graph())
        writer.flush()

    def train(self, x_train, y_train, ind2word):
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

        m_samples = np.shape(x_train)[0]
        total_batch = m_samples // FLAGS.batch_size
        loss_ = []
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, FLAGS.epochs + 1):
            loss_epoch = 0.0
            for batch in range(total_batch):
                sys.stdout.write('\r>>Epoch %d/%d | %d/%d' % (epoch, FLAGS.epochs, batch + 1, total_batch))
                sys.stdout.flush()

                x_batch = x_train[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size]
                y_batch = y_train[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size]
                loss_batch, _ = sess.run([self.loss, self.train_op],
                                         feed_dict={self.train_inputs: x_batch, self.train_labels: y_batch})
                loss_epoch += loss_batch
            loss_.append(loss_epoch / total_batch)
            sys.stdout.write(' | Loss:%.9f\n' % loss_[-1])

            r = np.random.permutation(m_samples)
            x_train = x_train[r]
            y_train = y_train[r]

            final_matrix = sess.run(self.norm_matrix)[1:FLAGS.plot_total + 1]
            low_dim_matrix = tsne.fit_transform(final_matrix)

            fig = plt.figure(figsize=(10, 10))
            for i in range(FLAGS.plot_total):
                plt.plot(low_dim_matrix[i, 0], low_dim_matrix[i, 1], 'r*')
                plt.text(low_dim_matrix[i, 0], low_dim_matrix[i, 1], ind2word[i + 1])
            plt.savefig(FLAGS.cbow_save, bbox_inches='tight')
            plt.close(fig)

        fig = plt.figure(figsize=(10, 10))
        plt.plot(loss_)
        plt.savefig(FLAGS.loss_save, bbox_inches='tight')
        plt.close(fig)


def main(unused_argv):
    lang = Lang()
    cbow = CBOW()
    cbow.train(lang.x_train, lang.y_train, lang.ind2word)


if __name__ == '__main__':
    tf.app.run()
