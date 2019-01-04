import tensorflow as tf
import numpy as np
import pickle
import os
import sys

tf.flags.DEFINE_integer('batch_size', 100, 'batch size for training')
tf.flags.DEFINE_integer('epochs', 200000, 'number of iterations')
tf.flags.DEFINE_integer('embedding_size', 50, 'embedding size for word embedding')
tf.flags.DEFINE_integer('rnn_size', 128, 'units of rnn')
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
        with tf.name_scope('embedding'):
            embedding_matrx = tf.Variable(
                tf.random_uniform([self.word_dict_len, self.config.embedding_size], -1.0, 1.0),
                dtype=tf.float32)
            embedded_sen = tf.nn.embedding_lookup(embedding_matrx, sen)

        with tf.name_scope('birnn'):
            rnn_fw = tf.nn.rnn_cell.GRUCell(self.config.rnn_size, name='fw')
            rnn_bw = tf.nn.rnn_cell.GRUCell(self.config.rnn_size, name='bw')

            birnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_fw,
                                                               rnn_bw,
                                                               embedded_sen,
                                                               sequence_length=length,
                                                               dtype=tf.float32)
            fb_outputs = tf.concat(birnn_outputs, 2)

            logits = tf.reshape(
                tf.layers.dense(tf.reshape(fb_outputs, [-1, 2 * self.config.rnn_size]), self.tagger_dict_len),
                [tf.shape(sen)[0], -1, self.tagger_dict_len])

        with tf.name_scope('loss'):
            prediction = tf.argmax(logits, axis=-1, name='prediction', output_type=tf.int32)
            j = tf.constant(0, tf.int32)
            k = tf.constant(0, tf.float32)

            def cond(j, k):
                return tf.less(j, tf.shape(sen)[0])

            def body(j, k):
                k += tf.reduce_sum(tf.cast(tf.equal(prediction[j, :length[j]], tag[j, :length[j]]), tf.float32))
                return tf.add(j, 1), k

            _, k = tf.while_loop(cond, body, [j, k])
            accuracy = tf.div(k, tf.cast(tf.reduce_sum(length), tf.float32), name='accuracy')

            masks = tf.sequence_mask(length,
                                     tf.reduce_max(length),
                                     dtype=tf.float32,
                                     name='masks')

            loss = tf.identity(tf.contrib.seq2seq.sequence_loss(logits, tag, masks), name='loss')
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr, name='optimizer')
            gradients = optimizer.compute_gradients(loss)
            clipped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(clipped_gradients, name='train_op')
        writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph(), filename_suffix='mypt1')
        writer.flush()
        writer.close()
        print('Graph saved successfully!')

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(sess, self.config.model_save_path + 'mypt1')
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

        newsaver = tf.train.import_meta_graph(self.config.model_save_path + 'mypt1.meta')
        newsaver.restore(sess, self.config.model_save_path + 'mypt1')

        graph = tf.get_default_graph()
        sen = graph.get_operation_by_name('input/sentences').outputs[0]
        tag = graph.get_operation_by_name('input/taggers').outputs[0]
        length = graph.get_operation_by_name('input/length').outputs[0]

        accuracy = graph.get_tensor_by_name('loss/accuracy:0')
        loss = graph.get_tensor_by_name('loss/loss:0')
        train_op = graph.get_operation_by_name('loss/train_op')

        loss_ = []
        acc_ = []
        for epoch in range(1, self.config.epochs + 1):
            train_batch_ = sess.run(train_batch)

            feed_dict = {sen: train_batch_[0],
                         tag: train_batch_[1],
                         length: train_batch_[2]
                         }
            loss_batch, _, acc_batch = sess.run([loss, train_op, accuracy], feed_dict=feed_dict)
            loss_.append(loss_batch)
            acc_.append(acc_batch)

            sys.stdout.write('\r>> %d/%d  | loss_batch: %f  acc_batch:%.3f' % (
                epoch, self.config.epochs, loss_batch, acc_batch))
            sys.stdout.flush()

            if epoch % self.config.per_save == 0:
                valid_acc = 0
                k = 0
                for i in range(20):
                    valid_batch_ = sess.run(valid_batch)

                    valid_feed_dict = {sen: valid_batch_[0],
                                       tag: valid_batch_[1],
                                       length: valid_batch_[2]
                                       }
                    tmp = np.sum(valid_batch_[2])
                    k += tmp
                    valid_acc += round(sess.run(accuracy, feed_dict=valid_feed_dict) * tmp)

                sys.stdout.write('  valid_acc:%.5f\n' % (valid_acc / k))
                sys.stdout.flush()

                newsaver.save(sess, self.config.model_save_path + 'mypt1')
                print('model saved successfully!')

        sess.close()

        fig = plt.figure(figsize=(10, 8))
        plt.plot(loss_)
        plt.savefig(self.config.model_save_path + 'mypt1_loss.png')
        plt.close(fig)

    def predict(self):
        test_file = ['data/test.tfrecord']

        test_batch = batched_data(test_file, single_example_parser, self.config.batch_size,
                                  padded_shapes=([-1], [-1], []),
                                  shuffle=False)
        sess = tf.Session()

        newsaver = tf.train.import_meta_graph(self.config.model_save_path + 'mypt1.meta')
        newsaver.restore(sess, self.config.model_save_path + 'mypt1')

        graph = tf.get_default_graph()
        sen = graph.get_operation_by_name('input/sentences').outputs[0]
        tag = graph.get_operation_by_name('input/taggers').outputs[0]
        length = graph.get_operation_by_name('input/length').outputs[0]

        accuracy = graph.get_tensor_by_name('loss/accuracy:0')

        test_batch_ = sess.run(test_batch)

        feed_dict = {sen: test_batch_[0],
                     tag: test_batch_[1],
                     length: test_batch_[2]
                     }
        acc_batch = sess.run(accuracy, feed_dict=feed_dict)
        print('\033[1;31;40m')
        sys.stdout.write('      test_acc:%.3f\n' % (acc_batch))
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
