'''
    My own Language Model for char
    完全并行
'''

import tensorflow as tf
import numpy as np
import pickle
import os
import sys
import matplotlib.pylab as plt
import jieba
from chardict import load_vocab

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽警告信息

tf.flags.DEFINE_integer('maxword', 512, 'max length of any sentences')
tf.flags.DEFINE_integer('block', 2, 'number of Encoder submodel')
tf.flags.DEFINE_integer('head', 12, 'number of multi_head attention')

tf.flags.DEFINE_integer('batch_size', 8, 'batch size for training')
tf.flags.DEFINE_integer('epochs', 20000000, 'number of iterations')
tf.flags.DEFINE_integer('embedding_size', 768, 'embedding size for word embedding')
tf.flags.DEFINE_string('model_save_path', 'model/zlm1/', 'directory of model file saved')
tf.flags.DEFINE_float('lr', 0.001, 'learning rate for training')
tf.flags.DEFINE_integer('pri', 256, 'topk choices')
tf.flags.DEFINE_float('keep_prob', 0.9, 'rate for dropout')
tf.flags.DEFINE_integer('per_save', 1000, 'save model once every per_save iterations')
tf.flags.DEFINE_string('mode', 'train0', 'The mode of train or predict as follows: '
                                         'train0: train first time or retrain'
                                         'train1: continue train'
                                         'detect: error detect')

CONFIG = tf.flags.FLAGS
PRIORITY = 200  # 检测原字出现的概率的排序前N位


def single_example_parser(serialized_example):
    context_features = {
        'ls': tf.FixedLenFeature([], tf.int64)
    }
    sequence_features = {
        'sen': tf.FixedLenSequenceFeature([], tf.int64)
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    ls = context_parsed['ls']

    sen = sequence_parsed['sen']
    return sen, ls


def batched_data(tfrecord_filename, single_example_parser, batch_size, padded_shapes, buffer_size=1000):
    dataset = tf.data.TFRecordDataset(tfrecord_filename) \
        .map(single_example_parser) \
        .shuffle(buffer_size) \
        .repeat() \
        .padded_batch(batch_size, padded_shapes=padded_shapes)

    return dataset.make_one_shot_iterator().get_next()


def layer_norm_compute(x, scale, bias, epsilon=1.0e-6):
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
    Af = tf.ones_like(A) * (1. - tf.pow(2., 31.))
    A = tf.where(Mask, A, Af)
    C = tf.nn.softmax(A, axis=-1)
    return C


def gelu(input_tensor):
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


class ZLM():
    def __init__(self, config, char_dict):
        self.config = config
        self.char_dict = char_dict
        self.char_dict_len = len(char_dict)
        self.char_reverse_dict = {v: k for k, v in char_dict.items()}

    def build_model(self):
        with tf.name_scope('input'):
            sen = tf.placeholder(tf.int32, [None, None], name='sen')
            sennow = tf.concat([sen, sen], axis=0)
            ls = tf.placeholder(tf.int32, [None], name='ls')
            max_ls = tf.reduce_max(ls)
            pos = tf.placeholder(tf.float32, [None, None], name='pos')
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            pri = tf.placeholder(tf.int32, name='pri')  # 提取前pri个可能的预测项

            padding_mask = tf.tile(tf.expand_dims(tf.greater(sennow, 0), 1),
                                   [self.config.head, tf.shape(sen)[1], 1])
            sequence_mask = tf.sequence_mask(ls)

            future_mask = tf.tile(tf.expand_dims(tf.sequence_mask(tf.range(1, limit=tf.shape(sen)[1] + 1)), 0),
                                  [tf.shape(sen)[0], 1, 1])
            past_mask = tf.transpose(future_mask, perm=[0, 2, 1])
            mask = tf.tile(tf.concat([future_mask, past_mask], axis=0), [self.config.head, 1, 1])
            mask_final = tf.logical_and(mask, padding_mask)

        with tf.name_scope('embedding'):
            embedding_matrx = tf.Variable(
                tf.random_uniform([self.char_dict_len, self.config.embedding_size], -1.0, 1.0),
                dtype=tf.float32)
            embedded_sen = tf.nn.embedding_lookup(embedding_matrx, sennow)
            scale_emb = tf.get_variable('scale_emb',
                                        initializer=tf.ones([self.config.embedding_size], dtype=tf.float32))
            bias_emb = tf.get_variable('bias_emb',
                                       initializer=tf.zeros([self.config.embedding_size], dtype=tf.float32))

            now = tf.nn.dropout(layer_norm_compute(embedded_sen + pos[:max_ls + 2], scale_emb, bias_emb), keep_prob)

        for block in range(self.config.block):
            with tf.variable_scope('selfattention' + str(block)):
                WQ = tf.layers.Dense(self.config.embedding_size, use_bias=False, name='Q')
                WK = tf.layers.Dense(self.config.embedding_size, use_bias=False, name='K')
                WV = tf.layers.Dense(self.config.embedding_size, use_bias=False, name='V')
                WO = tf.layers.Dense(self.config.embedding_size, use_bias=False, name='O')
                scale_sa = tf.get_variable('scale_sa',
                                           initializer=tf.ones([self.config.embedding_size], dtype=tf.float32))
                bias_sa = tf.get_variable('bias_sa',
                                          initializer=tf.zeros([self.config.embedding_size], dtype=tf.float32))
                # ------------------------------------------------------------------------------------------------------
                Q = tf.concat(tf.split(WQ(now), self.config.head, axis=-1), axis=0)
                K = tf.concat(tf.split(WK(now), self.config.head, axis=-1), axis=0)
                V = tf.concat(tf.split(WV(now), self.config.head, axis=-1), axis=0)
                QK = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(self.config.embedding_size / self.config.head)
                Z = WO(tf.concat(
                    tf.split(tf.matmul(tf.nn.dropout(softmax(QK, mask_final), keep_prob), V), self.config.head,
                             axis=0), axis=-1))
                now = tf.nn.dropout(layer_norm_compute(now + Z, scale_sa, bias_sa), keep_prob)

            with tf.variable_scope('feedforward' + str(block)):
                ffrelu = tf.layers.Dense(4 * self.config.embedding_size, activation=gelu, name='ffrelu')
                ff = tf.layers.Dense(self.config.embedding_size, name='ff')
                scale_ff = tf.get_variable('scale_ff',
                                           initializer=tf.ones([self.config.embedding_size], dtype=tf.float32))
                bias_ff = tf.get_variable('bias_ff',
                                          initializer=tf.zeros([self.config.embedding_size], dtype=tf.float32))
                now = tf.nn.dropout(layer_norm_compute(ff(ffrelu(now)) + now, scale_ff, bias_ff), keep_prob)

        with tf.variable_scope('project'):
            now1, now2 = tf.split(now, 2)
            now = now1[:, :-2] + now2[:, 2:]
            scale_pr = tf.get_variable('scale_pr',
                                       initializer=tf.ones([self.config.embedding_size], dtype=tf.float32))
            bias_pr = tf.get_variable('bias_pr',
                                      initializer=tf.zeros([self.config.embedding_size], dtype=tf.float32))
            now = layer_norm_compute(tf.layers.dense(now, self.config.embedding_size, activation=gelu), scale_pr,
                                     bias_pr)
            project_bias = tf.get_variable('project_bias', [self.char_dict_len],
                                           initializer=tf.zeros_initializer())

            logits = tf.matmul(tf.reshape(now, [-1, self.config.embedding_size]), embedding_matrx,
                               transpose_b=True) + project_bias
            logits = tf.reshape(logits, [-1, max_ls, self.char_dict_len])

        with tf.name_scope('loss'):
            prediction = tf.argmax(logits, axis=-1, output_type=tf.int32)
            predictionf = tf.zeros_like(prediction)
            prediction = tf.where(sequence_mask, prediction, predictionf)

            accuracy = tf.cast(tf.equal(prediction, sen[:, 1:-1]), tf.float32)
            accuracyf = tf.zeros_like(accuracy)
            accuracy = tf.div(tf.reduce_sum(tf.where(sequence_mask, accuracy, accuracyf)),
                              tf.cast(tf.reduce_sum(ls), tf.float32), name='accuracy')

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sen[:, 1:-1], logits=logits)
            lossf = tf.zeros_like(loss)
            loss = tf.reduce_mean(
                tf.div(tf.reduce_sum(tf.where(sequence_mask, loss, lossf), axis=-1), tf.cast(ls, tf.float32)),
                name='loss')

            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
            train_op = optimizer.minimize(loss, name='train_op')

            sort = tf.argsort(logits, axis=-1, direction='DESCENDING')[:, :, :pri]
            err = tf.reduce_sum(
                tf.cast(tf.equal(sort, tf.tile(tf.expand_dims(sen, axis=2), [1, 1, pri])[:, 1:-1]), tf.int32),
                axis=-1, name='err')

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

    def train(self, train_file):
        self.build_model()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=1)
        if self.config.mode == 'train1':
            saver.restore(sess, self.config.model_save_path)

        train_batch = batched_data(train_file, single_example_parser, self.config.batch_size,
                                   padded_shapes=([-1], []))

        graph = tf.get_default_graph()
        sen = graph.get_operation_by_name('input/sen').outputs[0]
        ls = graph.get_operation_by_name('input/ls').outputs[0]
        pos = graph.get_operation_by_name('input/pos').outputs[0]
        keep_prob = graph.get_operation_by_name('input/keep_prob').outputs[0]
        pri = graph.get_operation_by_name('input/pri').outputs[0]

        err = graph.get_tensor_by_name('loss/err:0')
        accuracy = graph.get_tensor_by_name('loss/accuracy:0')
        loss = graph.get_tensor_by_name('loss/loss:0')
        train_op = graph.get_operation_by_name('loss/train_op')

        pos_enc = np.array(
            [[position / np.power(10000.0, 2.0 * (i // 2) / self.config.embedding_size) for i in
              range(self.config.embedding_size)]
             for position in range(self.config.maxword)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])

        sentences = [
            '新中国的建立结束了的中国一百多年来任人宰割的历史。',
            '这是种过人民长期以来追求的的立想。',
            '证见会要切实维护的光大中小股动的核发权益。',
            '建立统亿的国家是种过人民长期以追求的的立项。',
            '深沪交易将当天涨跌幅度炒锅7%的股票都称为异动股。',
            '对于一个数据汾西方面的工作者来说,最熟的语言无疑就是python了。',
            '恰好我在像木中就遇到了这个问题,需要在Java程序中调用Python程序。',
            '我们基于机器学习构建了几个用于语音识别后文本的检错纠错模型。',
            '现任中央财经大学研究员，同时兼任苏州工业园区凌志软件股份有限公司独立董事、国建新能科技股份有限公司董事。'
        ]

        m_samples = len(sentences)

        sent = []
        leng = []
        for sentence in sentences:
            sen2id = [self.char_dict['[CLS]']] + [
                self.char_dict[word] if word in self.char_dict.keys() else self.char_dict['[UNK]']
                for word in sentence] + [self.char_dict['[SEP]']]
            sent.append(sen2id)
            leng.append(len(sentence))

        max_len = np.max(leng)
        for i in range(m_samples):
            if leng[i] < max_len:
                pad = [self.char_dict['[PAD]']] * (max_len - leng[i])
                sent[i] += pad

        feed_dict_test = {sen: sent,
                          ls: leng,
                          pos: pos_enc,
                          keep_prob: 1.0,
                          pri: self.config.pri
                          }

        # ------------------------------------------------------------------------------------------------------------------

        loss_ = []
        acc_ = []
        for epoch in range(1, self.config.epochs + 1):
            train_batch_ = sess.run(train_batch)

            feed_dict = {sen: train_batch_[0],
                         ls: train_batch_[1],
                         pos: pos_enc,
                         keep_prob: self.config.keep_prob
                         }
            loss_batch, acc_batch, _ = sess.run([loss, accuracy, train_op], feed_dict=feed_dict)
            loss_.append(loss_batch)
            acc_.append(acc_batch)

            sys.stdout.write('\r>> %d/%d  | loss_batch: %f  acc_batch:%.3f%%' % (
                epoch, self.config.epochs, loss_batch, 100.0 * acc_batch))
            sys.stdout.flush()

            if epoch % self.config.per_save == 0:
                sys.stdout.write('  train_loss:%f  train_acc:%.3f%%\n' % (
                    np.mean(loss_[-self.config.per_save:]),
                    100.0 * np.mean(acc_[-self.config.per_save:])))
                sys.stdout.flush()

                saver.save(sess, self.config.model_save_path)
                print('model saved successfully!')

                err_ = sess.run(err, feed_dict=feed_dict_test)

                # 检错环节
                for i in range(m_samples):
                    sentence = sentences[i]
                    error = err_[i]
                    words = jieba.lcut(sentence)
                    kwords = len(words)
                    ktext = len(sentence)

                    l1 = -1
                    for k in range(kwords):
                        lw = len(words[k])
                        if lw == 1:
                            l1 += 1
                        else:
                            for kk in range(l1 + 1, l1 + lw + 1):
                                if error[kk] == 0:
                                    error[l1 + 1:l1 + lw + 1] = 0
                                    break
                            l1 += lw

                    sys.stdout.write('检错: ')
                    for l in range(ktext):
                        if error[l] == 0:
                            sys.stdout.write('\033[4;31m%s\033[0m' % sentence[l])
                        else:
                            sys.stdout.write(sentence[l])
                    sys.stdout.write('\n\n')
                    sys.stdout.flush()

        sess.close()

        fig = plt.figure(figsize=(10, 8))
        plt.plot(loss_)
        plt.savefig(self.config.model_save_path)
        plt.close(fig)

    def detect(self, sentences):
        self.build_model()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=1)
        saver.restore(sess, self.config.model_save_path)

        graph = tf.get_default_graph()
        sen = graph.get_operation_by_name('input/sen').outputs[0]
        ls = graph.get_operation_by_name('input/ls').outputs[0]
        pos = graph.get_operation_by_name('input/pos').outputs[0]
        keep_prob = graph.get_operation_by_name('input/keep_prob').outputs[0]
        pri = graph.get_operation_by_name('input/pri').outputs[0]

        err = graph.get_tensor_by_name('loss/err:0')

        pos_enc = np.array(
            [[position / np.power(10000.0, 2.0 * (i // 2) / self.config.embedding_size) for i in
              range(self.config.embedding_size)]
             for position in range(self.config.maxword)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])

        m_samples = len(sentences)

        sent = []
        leng = []
        for sentence in sentences:
            sen2id = [self.char_dict['[CLS]']] + [
                self.char_dict[word] if word in self.char_dict.keys() else self.char_dict['[UNK]']
                for word in sentence] + [self.char_dict['[SEP]']]
            sent.append(sen2id)
            leng.append(len(sentence))

        max_len = np.max(leng)
        for i in range(m_samples):
            if leng[i] < max_len:
                pad = [self.char_dict['[PAD]']] * (max_len - leng[i])
                sent[i] += pad

        feed_dict = {
            sen: sent,
            ls: leng,
            pos: pos_enc,
            keep_prob: 1.0,
            pri: self.config.pri
        }

        err_ = sess.run(err, feed_dict=feed_dict)
        # 检错环节
        for i in range(m_samples):
            sentence = sentences[i]
            error = err_[i]
            words = jieba.lcut(sentence)
            kwords = len(words)
            ktext = len(sentence)

            l1 = -1
            for k in range(kwords):
                lw = len(words[k])
                if lw == 1:
                    l1 += 1
                else:
                    for kk in range(l1 + 1, l1 + lw + 1):
                        if error[kk] == 0:
                            error[l1 + 1:l1 + lw + 1] = 0
                            break
                    l1 += lw

            sys.stdout.write('检错: ')
            for l in range(ktext):
                if error[l] == 0:
                    sys.stdout.write('\033[4;31m%s\033[0m' % sentence[l])
                else:
                    sys.stdout.write(sentence[l])
            sys.stdout.write('\n\n')
            sys.stdout.flush()


def main(unused_argv):
    char_dict = load_vocab('vocab.txt')

    zlm = ZLM(CONFIG, char_dict)
    if CONFIG.mode == 'train0':
        if not os.path.exists(CONFIG.model_save_path):
            os.makedirs(CONFIG.model_save_path)
        zlm.train([
            '../data/train_finance.tfrecord',
            '../data/train_wiki.tfrecord',
        ])
    elif CONFIG.mode == 'train1':
        zlm.train([
            '../data/train_finance.tfrecord',
            '../data/train_wiki.tfrecord',
        ])
    elif CONFIG.mode == 'detect':
        sentences = [
            '新中国的建立结束了的中国一百多年来任人宰割的历史。',
            '这是种过人民长期以来追求的的立想。',
            '证见会要切实维护的光大中小股动的核发权益。',
            '建立统亿的国家是种过人民长期以追求的的立项。',
            '深沪交易将当天涨跌幅度炒锅7%的股票都称为异动股。',
            '对于一个数据汾西方面的工作者来说,最熟的语言无疑就是python了。',
            '恰好我在像木中就遇到了这个问题,需要在Java程序中调用Python程序。',
            '我们基于机器学习构建了几个用于语音识别后文本的检错纠错模型。',
            '现任中央财经大学研究员，同时兼任苏州工业园区凌志软件股份有限公司独立董事、国建新能科技股份有限公司董事。'
        ]
        zlm.detect(sentences)


if __name__ == '__main__':
    tf.app.run()
