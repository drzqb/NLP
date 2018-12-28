import tensorflow as tf
import numpy as np
import csv
from cbr_hp import CONFIG
from nltk.tokenize import WordPunctTokenizer
import collections
import sys
import os
import matplotlib.pylab as plt

class Lang():
    def __init__(self, config):
        self.config = config

        self.read_data()

        self.create_dict()

        self.train_word2ind()

        self.word2ind()

    def read_data(self):
        self.train = self.read_train_csv('data/train.csv')
        self.valid = self.read_csv('data/valid.csv')
        self.test = self.read_csv('data/test.csv')

    def read_csv(self, filename):
        tmp = []
        with open(filename) as f:
            csv_reader = csv.reader(f)
            title_reader = next(csv_reader)
            k = 1
            for row in csv_reader:
                if k <= 1000:
                    tmp.append([WordPunctTokenizer().tokenize(row[i].lower().strip(' ')) for i in range(len(row))])
                else:
                    break
                k += 1
        return tmp

    def read_train_csv(self, filename):
        tmp = []
        with open(filename) as f:
            csv_reader = csv.reader(f)
            title_reader = next(csv_reader)
            k = 1
            for row in csv_reader:
                if k <= 100000:
                    tmp.append([WordPunctTokenizer().tokenize(row[0].lower().strip(' ')),
                                WordPunctTokenizer().tokenize(row[1].lower().strip(' ')), row[2]])
                else:
                    break
                k += 1
        return tmp

    def create_dict(self):
        self.word_dict = {}

        tmp = []
        for sentences in self.train:
            tmp.extend(sentences[0])
            tmp.extend(sentences[1])

        self.word_dict['<unknown>'] = len(self.word_dict)

        counter = collections.Counter(tmp).most_common()

        for word, _ in counter:
            self.word_dict[word] = len(self.word_dict)

    def word2ind(self):
        self.valid_context = []
        self.valid_utterance = []
        self.valid_context_len = []
        self.valid_utterance_len = []
        self.valid_label = []

        m_samples = len(self.valid)
        for i in range(m_samples):
            tmp = self.valid[i]
            for j in range(1, 11):
                self.valid_context.append(
                    [self.word_dict[word] if word in self.word_dict.keys() else self.word_dict['<unknown>'] for word in
                     tmp[0]])
                self.valid_utterance.append(
                    [self.word_dict[word] if word in self.word_dict.keys() else self.word_dict['<unknown>'] for word in
                     tmp[j]])

                if j == 1:
                    self.valid_label.append(1.0)
                else:
                    self.valid_label.append(0.0)
                self.valid_context_len.append(len(tmp[0]))
                self.valid_utterance_len.append(len(tmp[j]))

        self.test_context = []
        self.test_utterance = []
        self.test_context_len = []
        self.test_utterance_len = []
        self.test_label = []

        m_samples = len(self.test)
        for i in range(m_samples):
            tmp = self.test[i]
            for j in range(1, 11):
                self.test_context.append(
                    [self.word_dict[word] if word in self.word_dict.keys() else self.word_dict['<unknown>'] for word in
                     tmp[0]])
                self.test_utterance.append(
                    [self.word_dict[word] if word in self.word_dict.keys() else self.word_dict['<unknown>'] for word in
                     tmp[j]])

                if j == 1:
                    self.test_label.append(1.0)
                else:
                    self.test_label.append(0.0)

                self.test_context_len.append(len(tmp[0]))
                self.test_utterance_len.append(len(tmp[j]))

    def train_word2ind(self):
        self.train_context = []
        self.train_utterance = []
        self.train_label = []

        m_samples = len(self.train)
        for i in range(m_samples):
            tmp = self.train[i]
            self.train_context.append(
                [self.word_dict[word] if word in self.word_dict.keys() else self.word_dict['<unknown>'] for word in
                 tmp[0]])
            self.train_utterance.append(
                [self.word_dict[word] if word in self.word_dict.keys() else self.word_dict['<unknown>'] for word in
                 tmp[1]])
            self.train_label.append(float(tmp[2]))
        self.train_context_len = [len(context) for context in self.train_context]
        self.train_utterance_len = [len(utterance) for utterance in self.train_utterance]


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

        writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph(), filename_suffix='cbr1')
        writer.flush()
        writer.close()
        print('graph written already!')

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=1)
        saver.save(sess, self.config.model_save_path + 'cbr1')
        sess.close()
        print('model saved already!')

    def train(self, lang):
        sess = tf.Session()

        newsaver = tf.train.import_meta_graph(self.config.model_save_path + 'cbr1.meta')
        newsaver.restore(sess, self.config.model_save_path + 'cbr1')

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

        m_samples = len(lang.train_context)
        total_batch = m_samples // self.config.batch_size

        valid_samples = len(lang.valid_context)
        valid_samples0 = len(lang.valid)

        loss_ = []
        acc_ = []
        for epoch in range(1, self.config.epochs + 1):
            loss_epoch = 0.0
            acc_epoch = 0.0
            for batch in range(total_batch):
                start_index = batch * self.config.batch_size
                end_index = (batch + 1) * self.config.batch_size

                context_batch = lang.train_context[start_index:end_index]
                utterance_batch = lang.train_utterance[start_index:end_index]
                context_len_batch = lang.train_context_len[start_index:end_index]
                utterance_len_batch = lang.train_utterance_len[start_index:end_index]
                label_batch = lang.train_label[start_index:end_index]

                context_batch = self.padding(context_batch, context_len_batch, 0)
                utterance_batch = self.padding(utterance_batch, utterance_len_batch, 0)

                feed_dict = {context: context_batch,
                             utterance: utterance_batch,
                             context_len: context_len_batch,
                             utterance_len: utterance_len_batch,
                             label: label_batch,
                             keep_prob: self.config.keep_prob}
                loss_batch, _, acc_batch = sess.run([loss, train_op, accuracy], feed_dict=feed_dict)
                loss_epoch += loss_batch
                acc_epoch += acc_batch

                sys.stdout.write('\r>> %d/%d | %d/%d  | loss_batch: %f  acc_batch:%f' % (
                    epoch, self.config.epochs, batch + 1, total_batch, loss_batch, acc_batch))
                sys.stdout.flush()

            loss_.append(loss_epoch / total_batch)
            acc_.append(acc_epoch / total_batch)
            sys.stdout.write(' | loss:%f  acc:%f\n' % (loss_[-1], acc_[-1]))
            sys.stdout.flush()

            valid_acc_ = 0.0
            k = np.zeros([len(self.config.top_k)], dtype=np.int32)
            for i in range(0, valid_samples, 10):
                start_index = i
                end_index = i + 10
                valid_context_batch = lang.valid_context[start_index:end_index]
                valid_utterance_batch = lang.valid_utterance[start_index:end_index]
                valid_context_len_batch = lang.valid_context_len[start_index:end_index]
                valid_utterance_len_batch = lang.valid_utterance_len[start_index:end_index]
                valid_label_batch = lang.valid_label[start_index:end_index]

                valid_utterance_batch = self.padding(valid_utterance_batch, valid_utterance_len_batch, 0)

                valid_feed_dict = {context: valid_context_batch,
                                   utterance: valid_utterance_batch,
                                   context_len: valid_context_len_batch,
                                   utterance_len: valid_utterance_len_batch,
                                   label: valid_label_batch,
                                   keep_prob: 1.0}

                valid_prob, valid_acc = sess.run([prob, accuracy], feed_dict=valid_feed_dict)
                valid_acc_ += valid_acc
                for j in range(len(self.config.top_k)):
                    if 0 in np.argsort(-valid_prob)[:self.config.top_k[j]]:
                        k[j] += 1

            sys.stdout.write('  valid_acc:%f' % (valid_acc_ / valid_samples0))
            for j in range(len(self.config.top_k)):
                sys.stdout.write(' | recall@(%d,10):%f' % (self.config.top_k[j], k[j] / valid_samples0))
            sys.stdout.write('\n')
            sys.stdout.flush()

            r = np.random.permutation(m_samples)
            lang.train_context = [lang.train_context[i] for i in r]
            lang.train_utterance = [lang.train_utterance[i] for i in r]
            lang.train_context_len = [lang.train_context_len[i] for i in r]
            lang.train_utterance_len = [lang.train_utterance_len[i] for i in r]
            lang.train_label = [lang.train_label[i] for i in r]

            if epoch % self.config.per_save == 0:
                newsaver.save(sess, self.config.model_save_path + 'cbr1')
                print('model saved successfully!')
        fig = plt.figure(figsize=(10, 8))
        plt.plot(loss_)
        plt.savefig(self.config.model_save_path + 'loss.png')
        plt.close(fig)

    def predict(self, lang):
        sess = tf.Session()

        newsaver = tf.train.import_meta_graph(self.config.model_save_path + 'cbr1.meta')
        newsaver.restore(sess, self.config.model_save_path + 'cbr1')

        graph = tf.get_default_graph()
        context = graph.get_operation_by_name('input/context').outputs[0]
        utterance = graph.get_operation_by_name('input/utterance').outputs[0]
        context_len = graph.get_operation_by_name('input/context_len').outputs[0]
        utterance_len = graph.get_operation_by_name('input/utterance_len').outputs[0]
        keep_prob = graph.get_operation_by_name('input/keep_prob').outputs[0]

        prob = graph.get_tensor_by_name('loss/prob:0')

        test_samples = len(lang.test_context)
        test_samples0 = len(lang.test)
        k = np.zeros([len(self.config.top_k)], dtype=np.int32)
        for i in range(0, test_samples, 10):
            start_index = i
            end_index = i + 10
            test_context_batch = lang.test_context[start_index:end_index]
            test_utterance_batch = lang.test_utterance[start_index:end_index]
            test_context_len_batch = lang.test_context_len[start_index:end_index]
            test_utterance_len_batch = lang.test_utterance_len[start_index:end_index]

            test_utterance_batch = self.padding(test_utterance_batch, test_utterance_len_batch, 0)

            test_feed_dict = {context: test_context_batch,
                              utterance: test_utterance_batch,
                              context_len: test_context_len_batch,
                              utterance_len: test_utterance_len_batch,
                              keep_prob: 1.0}

            test_prob = sess.run(prob, feed_dict=test_feed_dict)

            for j in range(len(self.config.top_k)):
                if 0 in np.argsort(-test_prob)[:self.config.top_k[j]]:
                    k[j] += 1

        for j in range(len(self.config.top_k)):
            sys.stdout.write('recall@(%d,10):%f  ' % (self.config.top_k[j], k[j] / test_samples0))
        sys.stdout.flush()

    def padding(self, x, l, padding_id):
        l_max = np.max(l)
        return [x[i] + [padding_id] * (l_max - l[i]) for i in range(len(x))]


def main(unused_argv):
    lang = Lang(CONFIG)
    print(len(lang.word_dict))

    cbr = chatbotretrieval(CONFIG, len(lang.word_dict))

    if CONFIG.mode == 'train0':
        if not os.path.exists(CONFIG.model_save_path):
            os.makedirs(CONFIG.model_save_path)
        cbr.build_model()
        cbr.train(lang)
    elif CONFIG.mode == 'train1':
        cbr.train(lang)
    elif CONFIG.mode == 'predict':
        cbr.predict(lang)


if __name__ == '__main__':
    tf.app.run()
