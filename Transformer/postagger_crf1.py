'''
    PosTagger with Transformer-CRF
'''

import tensorflow as tf
import numpy as np
import pickle
import os
import sys
import matplotlib.pylab as plt

tf.flags.DEFINE_integer('maxword', 1000, 'max length of any sentences')
tf.flags.DEFINE_integer('block', 2, 'number of Encoder submodel')
tf.flags.DEFINE_integer('head', 4, 'number of multi_head attention')

tf.flags.DEFINE_integer('batch_size', 100, 'batch size for training')
tf.flags.DEFINE_integer('epochs', 200000, 'number of iterations')
tf.flags.DEFINE_integer('embedding_size', 64, 'embedding size for word embedding')
tf.flags.DEFINE_string('model_save_path', 'model/', 'directory of model file saved')
tf.flags.DEFINE_float('lr', 0.001, 'learning rate for training')
tf.flags.DEFINE_float('keep_prob', 0.5, 'rate for dropout')
tf.flags.DEFINE_integer('per_save', 150, 'save model once every per_save iterations')
tf.flags.DEFINE_string('mode', 'train0', 'The mode of train or predict as follows: '
                                         'train0: train first time or retrain'
                                         'train1: continue train'
                                         'predict: predict')

CONFIG = tf.flags.FLAGS


def single_example_parser(serialized_example):
    context_features = {
        'length': tf.FixedLenFeature([], tf.int64)
    }
    sequence_features = {
        'sen': tf.FixedLenSequenceFeature([],
                                          tf.int64),
        'tag': tf.FixedLenSequenceFeature([],
                                          tf.int64)}

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    length = context_parsed['length']

    sen = sequence_parsed['sen']
    tag = sequence_parsed['tag']
    return sen, tag, length


def batched_data(tfrecord_filename, single_example_parser, batch_size, padded_shapes, buffer_size=1000,
                 shuffle=True):
    dataset = tf.data.TFRecordDataset(tfrecord_filename) \
        .map(single_example_parser) \
        .padded_batch(batch_size, padded_shapes=padded_shapes) \
        .repeat()
    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    return dataset.make_one_shot_iterator().get_next()


def layer_norm_compute(x, scale, bias, epsilon=1.0e-10):
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


class PosTagger():
    def __init__(self, config, word_dict_len, tagger_dict_len):
        self.config = config
        self.word_dict_len = word_dict_len
        self.tagger_dict_len = tagger_dict_len

    def build_model(self):
        with tf.name_scope('input'):
            sen = tf.placeholder(tf.int32, [None, None], name='sentences')
            tag = tf.placeholder(tf.int32, [None, None], name='taggers')
            length = tf.placeholder(tf.int32, [None], name='length')
            max_length = tf.reduce_max(length, name='max_length')
            pos = tf.placeholder(tf.float32, [None, None], name='position')
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            padding_mask = tf.tile(tf.expand_dims(tf.greater(sen, 0), 1), [self.config.head, tf.shape(sen)[1], 1])
            sequence_mask = tf.sequence_mask(length, max_length)

        with tf.name_scope('embedding'):
            transition_params = tf.Variable(tf.random_uniform([self.tagger_dict_len, self.tagger_dict_len], 0.0, 1.0),
                                            dtype=tf.float32)
            embedding_matrx = tf.Variable(
                tf.random_uniform([self.word_dict_len, self.config.embedding_size], -1.0, 1.0),
                dtype=tf.float32)
            embedded_sen = tf.nn.embedding_lookup(embedding_matrx, sen)
            now = tf.nn.dropout(embedded_sen + pos[:max_length], keep_prob)

        for block in range(self.config.block):
            with tf.variable_scope('selfattention' + str(block)):
                WQ = tf.layers.Dense(self.config.embedding_size, use_bias=False, name='Q')
                WK = tf.layers.Dense(self.config.embedding_size, use_bias=False, name='K')
                WV = tf.layers.Dense(self.config.embedding_size, use_bias=False, name='V')
                WO = tf.layers.Dense(self.config.embedding_size, use_bias=False, name='O')

                Q = tf.concat(tf.split(WQ(now), self.config.head, axis=-1), axis=0)
                K = tf.concat(tf.split(WK(now), self.config.head, axis=-1), axis=0)
                V = tf.concat(tf.split(WV(now), self.config.head, axis=-1), axis=0)

                QK = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(self.config.embedding_size / self.config.head)
                Z = tf.nn.dropout(WO(tf.concat(
                    tf.split(tf.matmul(softmax(QK, padding_mask), V), self.config.head,
                             axis=0), axis=-1)), keep_prob)
                scale_sa = tf.get_variable('scale_sa',
                                           initializer=tf.ones([self.config.embedding_size], dtype=tf.float32))
                bias_sa = tf.get_variable('bias_sa',
                                          initializer=tf.zeros([self.config.embedding_size], dtype=tf.float32))
                now = layer_norm_compute(now + Z, scale_sa, bias_sa)
            with tf.variable_scope('feedforward' + str(block)):
                ffrelu = tf.layers.Dense(4 * self.config.embedding_size, activation=tf.nn.relu, name='ffrelu')
                ff = tf.layers.Dense(self.config.embedding_size, name='ff')
                scale_ff = tf.get_variable('scale_ff',
                                           initializer=tf.ones([self.config.embedding_size], dtype=tf.float32))
                bias_ff = tf.get_variable('bias_ff',
                                          initializer=tf.zeros([self.config.embedding_size], dtype=tf.float32))
                now = layer_norm_compute(ff(ffrelu(now)) + now, scale_ff, bias_ff)

        with tf.variable_scope('project'):
            logits = tf.layers.dense(now, self.tagger_dict_len)

        with tf.name_scope('loss'):
            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, tag, length, transition_params)
            viterbi_sequence, _ = tf.contrib.crf.crf_decode(logits, transition_params, length)

            loss = tf.reduce_mean(-log_likelihood, name='loss')

            accuracy = tf.cast(tf.equal(viterbi_sequence, tag), tf.float32)
            accuracyf = tf.zeros_like(accuracy)
            accuracy = tf.div(tf.reduce_sum(tf.where(sequence_mask, accuracy, accuracyf)),
                              tf.cast(tf.reduce_sum(length), tf.float32), name='accuracy')

            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr, name='optimizer')
            train_op = optimizer.minimize(loss, name='train_op')
        writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph(), filename_suffix='mypt_crf2')
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

        saver = tf.train.Saver(max_to_keep=1)
        saver.save(sess, self.config.model_save_path + 'mypt_crf2')
        sess.close()
        print('Model saved successfully!')

    def train(self):
        train_file = ['data/train.tfrecord']
        valid_file = ['data/valid.tfrecord']

        train_batch = batched_data(train_file, single_example_parser, self.config.batch_size,
                                   padded_shapes=([-1], [-1], []))
        valid_batch = batched_data(valid_file, single_example_parser, self.config.batch_size,
                                   padded_shapes=([-1], [-1], []),
                                   shuffle=False)
        sess = tf.Session()

        newsaver = tf.train.import_meta_graph(self.config.model_save_path + 'mypt_crf2.meta')
        newsaver.restore(sess, self.config.model_save_path + 'mypt_crf2')

        graph = tf.get_default_graph()
        sen = graph.get_operation_by_name('input/sentences').outputs[0]
        tag = graph.get_operation_by_name('input/taggers').outputs[0]
        length = graph.get_operation_by_name('input/length').outputs[0]
        pos = graph.get_operation_by_name('input/position').outputs[0]
        keep_prob = graph.get_operation_by_name('input/keep_prob').outputs[0]

        accuracy = graph.get_tensor_by_name('loss/accuracy:0')
        loss = graph.get_tensor_by_name('loss/loss:0')
        train_op = graph.get_operation_by_name('loss/train_op')

        pos_enc = np.array(
            [[position / np.power(10000.0, 2.0 * (i // 2) / self.config.embedding_size) for i in
              range(self.config.embedding_size)]
             for position in range(self.config.maxword)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])

        loss_ = []
        acc_ = []
        for epoch in range(1, self.config.epochs + 1):
            train_batch_ = sess.run(train_batch)

            feed_dict = {sen: train_batch_[0],
                         tag: train_batch_[1],
                         length: train_batch_[2],
                         pos: pos_enc,
                         keep_prob: self.config.keep_prob
                         }
            loss_batch, acc_batch, _ = sess.run([loss, accuracy, train_op], feed_dict=feed_dict)
            loss_.append(loss_batch)
            acc_.append(acc_batch)

            sys.stdout.write('\r>> %d/%d  | loss_batch: %f  acc_batch:%.2f%%' % (
                epoch, self.config.epochs, loss_batch, 100.0 * acc_batch))
            sys.stdout.flush()

            if epoch % self.config.per_save == 0:
                valid_acc = 0
                k = 0
                for i in range(20):
                    valid_batch_ = sess.run(valid_batch)

                    valid_feed_dict = {sen: valid_batch_[0],
                                       tag: valid_batch_[1],
                                       length: valid_batch_[2],
                                       pos: pos_enc,
                                       keep_prob: 1.0
                                       }
                    tmp = np.sum(valid_batch_[2])
                    k += tmp
                    valid_acc += round(sess.run(accuracy, feed_dict=valid_feed_dict) * tmp)

                sys.stdout.write('  train_loss:%f  train_acc:%.2f%%  |  valid_acc:%.2f%%\n' % (
                    np.mean(loss_[-self.config.per_save:]),
                    100.0 * np.mean(acc_[-self.config.per_save:]),
                    100.0 * valid_acc / k))
                sys.stdout.flush()

                newsaver.save(sess, self.config.model_save_path + 'mypt_crf2')
                print('model saved successfully!')

        sess.close()

        fig = plt.figure(figsize=(10, 8))
        plt.plot(loss_)
        plt.savefig(self.config.model_save_path + 'mypt_crf2_loss.png')
        plt.close(fig)

    def predict(self):
        test_file = ['data/test.tfrecord']

        test_batch = batched_data(test_file, single_example_parser, self.config.batch_size,
                                  padded_shapes=([-1], [-1], []),
                                  shuffle=False)
        sess = tf.Session()

        newsaver = tf.train.import_meta_graph(self.config.model_save_path + 'mypt_crf2.meta')
        newsaver.restore(sess, self.config.model_save_path + 'mypt_crf2')

        graph = tf.get_default_graph()
        sen = graph.get_operation_by_name('input/sentences').outputs[0]
        tag = graph.get_operation_by_name('input/taggers').outputs[0]
        length = graph.get_operation_by_name('input/length').outputs[0]
        pos = graph.get_operation_by_name('input/position').outputs[0]
        keep_prob = graph.get_operation_by_name('input/keep_prob').outputs[0]

        accuracy = graph.get_tensor_by_name('loss/accuracy:0')
        pos_enc = np.array(
            [[position / np.power(10000.0, 2.0 * (i // 2) / self.config.embedding_size) for i in
              range(self.config.embedding_size)]
             for position in range(self.config.maxword)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])

        test_batch_ = sess.run(test_batch)

        feed_dict = {sen: test_batch_[0],
                     tag: test_batch_[1],
                     length: test_batch_[2],
                     pos: pos_enc,
                     keep_prob: 1.0
                     }
        acc_batch = sess.run(accuracy, feed_dict=feed_dict)
        print('\033[1;31;40m')
        sys.stdout.write('      test_acc:%.2f%%\n' % (100.0 * acc_batch))
        sys.stdout.flush()
        print('\033[0m')

        sess.close()


def main(unused_argv):
    with open('data/word_dict.txt', 'rb') as f:
        word_dict = pickle.load(f)
    with open('data/tagger_dict.txt', 'rb') as f:
        tagger_dict = pickle.load(f)

    postagger = PosTagger(CONFIG, len(word_dict), len(tagger_dict))
    if CONFIG.mode == 'train0':
        if not os.path.exists(CONFIG.model_save_path):
            os.makedirs(CONFIG.model_save_path)
        postagger.build_model()
        postagger.train()
    elif CONFIG.mode == 'train1':
        postagger.train()
    elif CONFIG.mode == 'predict':
        postagger.predict()


if __name__ == '__main__':
    tf.app.run()
