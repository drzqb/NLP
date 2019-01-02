import tensorflow as tf
import numpy as np
import sys
import os
import matplotlib.pylab as plt
import pickle
import math

tf.flags.DEFINE_integer('batch_size', 100, 'batch size for training')
tf.flags.DEFINE_integer('epochs', 200000, 'number of iterations')
tf.flags.DEFINE_integer('embedding_dize', 50, 'embedding size for word embedding')
tf.flags.DEFINE_integer('rnn_size', 100, 'units of rnn')
tf.flags.DEFINE_string('model_save_path', 'model/', 'directory of model file saved')
tf.flags.DEFINE_float('lr', 0.001, 'learning rate for training')
tf.flags.DEFINE_float('keep_prob', 0.5, 'rate for dropout')
tf.flags.DEFINE_multi_integer('top_k', [1, 2, 5], 'top top_k numbers of a vector')
tf.flags.DEFINE_integer('per_save', 2000, 'save model once every per_save iterations')
tf.flags.DEFINE_string('mode', 'train0', 'The mode of train or predict as follows: '
                                         'train0: train first time or retrain'
                                         'train1: continue train'
                                         'predict: predict')
CONFIG = tf.flags.FLAGS


def single_example_parser(serialized_example):
    context_features = {
        'label': tf.FixedLenFeature([], tf.float32),
        'context_len': tf.FixedLenFeature([], tf.int64),
        'utterance_len': tf.FixedLenFeature([], tf.int64)
    }
    sequence_features = {
        'context': tf.FixedLenSequenceFeature([],
                                              tf.int64),
        'utterance': tf.FixedLenSequenceFeature([],
                                                tf.int64)}

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    context = sequence_parsed['context']
    utterance = sequence_parsed['utterance']
    label = context_parsed['label']
    context_len = context_parsed['context_len']
    utterance_len = context_parsed['utterance_len']
    return context, utterance, context_len, utterance_len, label


def batched_data(tfrecord_filename, single_example_parser, batch_size, padded_shapes, num_epochs=10, buffer_size=1000,
                 shuffle=True):
    dataset = tf.data.TFRecordDataset(tfrecord_filename) \
        .map(single_example_parser) \
        .padded_batch(batch_size, padded_shapes=padded_shapes) \
        .repeat()
    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    return dataset.make_one_shot_iterator().get_next()


class chatbotretrieval():
    def __init__(self, config, dict_len):
        self.config = config
        self.dict_len = dict_len

    def build_model(self):
        with tf.name_scope('input'):
            context = tf.placeholder(tf.int32, [None, None], name='context')
            utterance = tf.placeholder(tf.int32, [None, None], name='utterance')
            context_len = tf.placeholder(tf.int32, [None], name='context_len')
            utterance_len = tf.placeholder(tf.int32, [None], name='utterance_len')
            label = tf.placeholder(tf.float32, [None], name='label')
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.name_scope('embedding'):
            embedding_matrx = tf.Variable(tf.random_uniform([self.dict_len, self.config.embedding_dize], -1.0, 1.0),
                                          dtype=tf.float32)
            embedded_context = tf.nn.embedding_lookup(embedding_matrx, context)
            embedded_utterance = tf.nn.embedding_lookup(embedding_matrx, utterance)

        with tf.variable_scope('rnn_context_utterance'):
            rnn_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(self.config.rnn_size),
                                                     output_keep_prob=keep_prob)
            _, context_final_state = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                       inputs=embedded_context,
                                                       sequence_length=context_len,
                                                       dtype=tf.float32)

            _, utterance_final_state = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                         inputs=embedded_utterance,
                                                         sequence_length=utterance_len,
                                                         dtype=tf.float32)
        with tf.name_scope('loss'):
            M = tf.Variable(
                tf.random_uniform([self.config.rnn_size, self.config.rnn_size], -1.0, 1.0),
                dtype=tf.float32)
            output = tf.squeeze(
                tf.matmul(tf.reshape(tf.matmul(context_final_state, M), [-1, 1, self.config.rnn_size]),
                          tf.reshape(utterance_final_state, [-1, self.config.rnn_size, 1])))
            prob = tf.sigmoid(output, name='prob')
            predict = tf.map_fn(lambda x: tf.cond(tf.less(x, 0.0), lambda: 0.0, lambda: 1.0), output, dtype=tf.float32,
                                name='predict')
            accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, label), tf.float32), name='accuracy')

            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=label), name='loss')
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
            train_op = optimizer.minimize(loss, name='train_op')

        writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph(), filename_suffix='mycbr1')
        writer.flush()
        writer.close()
        print('graph written already!')

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=1)
        saver.save(sess, self.config.model_save_path + 'mycbr1')
        sess.close()
        print('model saved already!')

    def train(self):
        train_file = ['data/train.tfrecord']
        valid_file = ['data/valid.tfrecord']

        train_batch = batched_data(train_file, single_example_parser, self.config.batch_size,
                                   padded_shapes=([-1], [-1], [], [], []))
        valid_batch = batched_data(valid_file, single_example_parser, 10, padded_shapes=([-1], [-1], [], [], []),
                                   shuffle=False)
        sess = tf.Session()

        newsaver = tf.train.import_meta_graph(self.config.model_save_path + 'mycbr1.meta')
        newsaver.restore(sess, self.config.model_save_path + 'mycbr1')

        graph = tf.get_default_graph()
        context = graph.get_operation_by_name('input/context').outputs[0]
        utterance = graph.get_operation_by_name('input/utterance').outputs[0]
        context_len = graph.get_operation_by_name('input/context_len').outputs[0]
        utterance_len = graph.get_operation_by_name('input/utterance_len').outputs[0]
        label = graph.get_operation_by_name('input/label').outputs[0]
        keep_prob = graph.get_operation_by_name('input/keep_prob').outputs[0]

        prob = graph.get_tensor_by_name('loss/prob:0')
        predict = graph.get_tensor_by_name('loss/predict/TensorArrayStack/TensorArrayGatherV3:0')
        accuracy = graph.get_tensor_by_name('loss/accuracy:0')
        loss = graph.get_tensor_by_name('loss/loss:0')
        train_op = graph.get_operation_by_name('loss/train_op')

        loss_ = []
        acc_ = []
        for epoch in range(1, self.config.epochs + 1):
            train_batch_ = sess.run(train_batch)

            feed_dict = {context: train_batch_[0],
                         utterance: train_batch_[1],
                         context_len: train_batch_[2],
                         utterance_len: train_batch_[3],
                         label: train_batch_[4],
                         keep_prob: self.config.keep_prob}
            loss_batch, _, acc_batch = sess.run([loss, train_op, accuracy], feed_dict=feed_dict)
            loss_.append(loss_batch)
            acc_.append(acc_batch)

            sys.stdout.write('\r>> %d/%d  | loss_batch: %f  acc_batch:%.3f' % (
                epoch, self.config.epochs, loss_batch, acc_batch))
            sys.stdout.flush()

            if epoch % self.config.per_save == 0:
                valid_acc_ = 0.0
                k = np.zeros([len(self.config.top_k)], dtype=np.int32)
                for i in range(0, 1000):
                    valid_batch_ = sess.run(valid_batch)

                    valid_feed_dict = {context: valid_batch_[0],
                                       utterance: valid_batch_[1],
                                       context_len: valid_batch_[2],
                                       utterance_len: valid_batch_[3],
                                       label: valid_batch_[4],
                                       keep_prob: 1.0}

                    valid_prob, valid_acc = sess.run([prob, accuracy], feed_dict=valid_feed_dict)
                    valid_acc_ += valid_acc
                    for j in range(len(self.config.top_k)):
                        if 0 in np.argsort(-valid_prob)[:self.config.top_k[j]]:
                            k[j] += 1

                sys.stdout.write('  valid_acc:%.3f' % (valid_acc_ / 1000))
                for j in range(len(self.config.top_k)):
                    sys.stdout.write(' | recall@(%d,10):%.3f' % (self.config.top_k[j], k[j] / 1000))
                sys.stdout.write('\n')
                sys.stdout.flush()

                newsaver.save(sess, self.config.model_save_path + 'mycbr1')
                print('model saved successfully!')

        sess.close()

        fig = plt.figure(figsize=(10, 8))
        plt.plot(loss_)
        plt.savefig(self.config.model_save_path + 'mycbr1_loss.png')
        plt.close(fig)

    def predict(self):
        test_file = ['data/test.tfrecord']

        test_batch = batched_data(test_file, single_example_parser, 10, padded_shapes=([-1], [-1], [], [], []),
                                  shuffle=False)
        sess = tf.Session()

        newsaver = tf.train.import_meta_graph(self.config.model_save_path + 'mycbr1.meta')
        newsaver.restore(sess, self.config.model_save_path + 'mycbr1')

        graph = tf.get_default_graph()
        context = graph.get_operation_by_name('input/context').outputs[0]
        utterance = graph.get_operation_by_name('input/utterance').outputs[0]
        context_len = graph.get_operation_by_name('input/context_len').outputs[0]
        utterance_len = graph.get_operation_by_name('input/utterance_len').outputs[0]
        keep_prob = graph.get_operation_by_name('input/keep_prob').outputs[0]

        prob = graph.get_tensor_by_name('loss/prob:0')

        k = np.zeros([len(self.config.top_k)], dtype=np.int32)
        for i in range(0, 1000):
            test_batch_ = sess.run(test_batch)
            test_feed_dict = {context: test_batch_[0],
                              utterance: test_batch_[1],
                              context_len: test_batch_[2],
                              utterance_len: test_batch_[3],
                              keep_prob: 1.0}

            test_prob = sess.run(prob, feed_dict=test_feed_dict)

            for j in range(len(self.config.top_k)):
                if 0 in np.argsort(-test_prob)[:self.config.top_k[j]]:
                    k[j] += 1

        print('\033[1;31;40m')

        for j in range(len(self.config.top_k)):
            sys.stdout.write('  recall@(%d,10):%f' % (self.config.top_k[j], k[j] / 1000))
        sys.stdout.write('\n')
        sys.stdout.flush()

        print('\033[0m')

        sess.close()


def main(unused_argv):
    with open('data/word_dict.txt', 'rb') as f:
        word_dict = pickle.load(f)

    cbr = chatbotretrieval(CONFIG, len(word_dict))

    if CONFIG.mode == 'train0':
        if not os.path.exists(CONFIG.model_save_path):
            os.makedirs(CONFIG.model_save_path)
        cbr.build_model()
        cbr.train()
    elif CONFIG.mode == 'train1':
        cbr.train()
    elif CONFIG.mode == 'predict':
        cbr.predict()


if __name__ == '__main__':
    tf.app.run()
