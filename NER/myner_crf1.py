'''
    BiLSTM-CRF for NER
'''
import tensorflow as tf
import numpy as np
import pickle
import os
import sys
import matplotlib.pylab as plt
from tensorflow.contrib.data import AUTOTUNE

tf.flags.DEFINE_integer('batch_size', 100, 'batch size for training')
tf.flags.DEFINE_integer('epochs', 200000, 'number of iterations')
tf.flags.DEFINE_integer('embedding_size', 50, 'embedding size for word embedding')
tf.flags.DEFINE_integer('rnn_size', 128, 'units of rnn')
tf.flags.DEFINE_string('model_save_path', 'model/nercrf1/', 'directory of model file saved')
tf.flags.DEFINE_float('lr', 0.001, 'learning rate for training')
tf.flags.DEFINE_float('keep_prob', 0.5, 'rate for dropout')
tf.flags.DEFINE_integer('per_save', 50, 'save model once every per_save iterations')
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
        'ner': tf.FixedLenSequenceFeature([],
                                          tf.int64)}

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    length = context_parsed['length']

    sen = sequence_parsed['sen']
    ner = sequence_parsed['ner']
    return sen, ner, length


def batched_data(tfrecord_filename, single_example_parser, batch_size, padded_shapes, buffer_size=1000,
                 shuffle=True):
    dataset = tf.data.TFRecordDataset(tfrecord_filename) \
        .map(single_example_parser) \
        .padded_batch(batch_size, padded_shapes=padded_shapes) \
        .repeat() \
        .prefetch(buffer_size=AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    return dataset.make_one_shot_iterator().get_next()


class NER():
    def __init__(self, config, word_dict_len, ner_dict_len):
        self.config = config
        self.word_dict_len = word_dict_len
        self.ner_dict_len = ner_dict_len

    def build_model(self):
        with tf.name_scope('input'):
            sen = tf.placeholder(tf.int32, [None, None], name='sentences')
            ner = tf.placeholder(tf.int32, [None, None], name='ners')
            length = tf.placeholder(tf.int32, [None], name='length')
        with tf.name_scope('embedding'):
            transition_params = tf.Variable(tf.random_uniform([self.ner_dict_len, self.ner_dict_len], 0.0, 1.0),
                                            dtype=tf.float32)

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
                tf.layers.dense(tf.reshape(fb_outputs, [-1, 2 * self.config.rnn_size]), self.ner_dict_len),
                [tf.shape(sen)[0], -1, self.ner_dict_len])

        with tf.name_scope('loss'):
            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, ner, length, transition_params)
            viterbi_sequence, _ = tf.contrib.crf.crf_decode(logits, transition_params, length)

            prediction = tf.identity(viterbi_sequence, name='prediction')
            loss = tf.reduce_mean(-log_likelihood, name='loss')

            masks = tf.sequence_mask(length, tf.reduce_max(length))
            accuracy = tf.cast(tf.equal(viterbi_sequence, ner), tf.float32)
            accuracyf = tf.zeros_like(accuracy)
            accuracy = tf.div(tf.reduce_sum(tf.where(masks, accuracy, accuracyf)),
                              tf.cast(tf.reduce_sum(length), tf.float32), name='accuracy')

            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr, name='optimizer')
            gradients = optimizer.compute_gradients(loss)
            clipped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(clipped_gradients, name='train_op')

        writer = tf.summary.FileWriter(self.config.model_save_path, graph=tf.get_default_graph())
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
        saver.save(sess, self.config.model_save_path)
        sess.close()
        print('Model saved successfully!')

    def train(self):
        train_file = ['data/train.tfrecord']
        valid_file = ['data/valid.tfrecord']

        train_batch = batched_data(train_file, single_example_parser, self.config.batch_size,
                                   padded_shapes=([-1], [-1], []))
        valid_batch = batched_data(valid_file, single_example_parser, 110,
                                   padded_shapes=([-1], [-1], []),
                                   shuffle=False)
        sess = tf.Session()

        newsaver = tf.train.import_meta_graph(self.config.model_save_path + '.meta')
        newsaver.restore(sess, self.config.model_save_path)

        graph = tf.get_default_graph()
        sen = graph.get_operation_by_name('input/sentences').outputs[0]
        ner = graph.get_operation_by_name('input/ners').outputs[0]
        length = graph.get_operation_by_name('input/length').outputs[0]

        accuracy = graph.get_tensor_by_name('loss/accuracy:0')
        loss = graph.get_tensor_by_name('loss/loss:0')
        train_op = graph.get_operation_by_name('loss/train_op')

        loss_ = []
        acc_ = []
        for epoch in range(1, self.config.epochs + 1):
            train_batch_ = sess.run(train_batch)

            feed_dict = {sen: train_batch_[0],
                         ner: train_batch_[1],
                         length: train_batch_[2]
                         }
            loss_batch, _, acc_batch = sess.run([loss, train_op, accuracy], feed_dict=feed_dict)
            loss_.append(loss_batch)
            acc_.append(acc_batch)

            sys.stdout.write('\r>> %d/%d  | loss_batch: %f  acc_batch:%.3f' % (
                epoch, self.config.epochs, loss_batch, acc_batch))
            sys.stdout.flush()

            if epoch % self.config.per_save == 0:
                valid_batch_ = sess.run(valid_batch)

                valid_feed_dict = {sen: valid_batch_[0],
                                   ner: valid_batch_[1],
                                   length: valid_batch_[2]
                                   }
                valid_acc = sess.run(accuracy, feed_dict=valid_feed_dict)

                sys.stdout.write('  train_loss: %f  train_acc: %.3f | valid_acc:%.3f\n'
                                 % (np.mean(loss_[-self.config.batch_size:]),
                                    np.mean(acc_[-self.config.batch_size:]),
                                    valid_acc))
                sys.stdout.flush()

                newsaver.save(sess, self.config.model_save_path)
                print('model saved successfully!')

        sess.close()

        fig = plt.figure(figsize=(10, 8))
        plt.plot(loss_)
        plt.savefig(self.config.model_save_path + 'loss.png')
        plt.close(fig)

    def predict(self, word_dict, ner_dict):
        ner_reverse_dict = {v: k for k, v in ner_dict.items()}

        sentences = [
            '第二十二届 国际 检察官 联合会 年会 暨 会员 代表大会 11 日 上午 在 北京 开幕 。 国家 主席 习近平 发来 贺信 ， 对 会议 召开 表示祝贺 。',
            '重庆市 江边 未建 投放 垃圾 的 设施 ， 居民 任意 向 江边 倒 脏物 。',
            '伪造 、 买卖 、 非法 提供 、 非法 使用 武装部队 专用 标志 罪'
        ]

        m_samples = len(sentences)

        sent = []
        leng = []
        for sentence in sentences:
            sen2id = [word_dict[word] if word in word_dict.keys() else word_dict['<unknown>'] for word in
                      sentence.split(' ')]
            sent.append(sen2id)
            leng.append(len(sen2id))

        max_len = np.max(leng)
        for i in range(m_samples):
            if leng[i] < max_len:
                sent[i] += [word_dict['<pad>']] * (max_len - leng[i])

        sess = tf.Session()

        newsaver = tf.train.import_meta_graph(self.config.model_save_path + '.meta')
        newsaver.restore(sess, self.config.model_save_path)

        graph = tf.get_default_graph()
        sen = graph.get_operation_by_name('input/sentences').outputs[0]
        length = graph.get_operation_by_name('input/length').outputs[0]

        prediction = graph.get_tensor_by_name('loss/prediction:0')

        feed_dict = {sen: sent,
                     length: leng
                     }
        prediction_ = sess.run(prediction, feed_dict=feed_dict)

        for i in range(m_samples):
            tmp = []
            for idx in prediction_[i]:
                tmp.append(ner_reverse_dict[idx])
            sys.stdout.write('SEN: %s\n' % (sentences[i]))
            sys.stdout.write('NER: %s\n\n' % (' '.join(tmp[:leng[i]])))
        sys.stdout.flush()

        sess.close()


def main(unused_argv):
    with open('data/word_dict.txt', 'rb') as f:
        word_dict = pickle.load(f)
    with open('data/ner_dict.txt', 'rb') as f:
        ner_dict = pickle.load(f)

    ner = NER(CONFIG, len(word_dict), len(ner_dict))
    if CONFIG.mode == 'train0':
        if not os.path.exists(CONFIG.model_save_path):
            os.makedirs(CONFIG.model_save_path)
        ner.build_model()
        ner.train()
    elif CONFIG.mode == 'train1':
        ner.train()
    elif CONFIG.mode == 'predict':
        ner.predict(word_dict, ner_dict)


if __name__ == '__main__':
    tf.app.run()
