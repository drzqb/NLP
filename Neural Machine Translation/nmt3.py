'''
    seq2seq with attention after gru layers
             and inference
             and two gru layers
             for neural translation
'''
import numpy as np
import tensorflow as tf
import sys, collections, re, os
import jieba, zhon.hanzi, string
from tensorflow.contrib import seq2seq
import matplotlib.pylab as plt
from nltk.tokenize import WordPunctTokenizer
from nmt_hparams import FLAGS


class Lang():
    '''
    读取中英文语料并建立中英文字典
    '''

    def __init__(self):
        if isinstance(FLAGS.corpus, list) or isinstance(FLAGS.corpus, tuple):
            self.read_raw_data()
        else:
            self.read_data()

        self.create_dict()
        self.text2id()

    # 读取中英文语料
    def read_data(self):
        self.raw_en = []
        self.raw_ch = []
        s = open(FLAGS.corpus, 'r', encoding='utf-8').read()
        s = s.split('\n')

        for sl in s[:]:
            tmp_sl = sl.strip(' ')
            tmp_sl = re.sub(' ', '', tmp_sl)
            tmp_sl = re.sub('\t', '', tmp_sl)
            if tmp_sl == '':
                s.remove(sl)

        for sl in s[1:len(s):2]:
            self.raw_en.append(WordPunctTokenizer().tokenize(sl.lower().strip(' ')))

        for sl in s[0:-1:2]:
            self.raw_ch.append(jieba.lcut(re.sub(' |\u3000', '', sl.strip(' '))))

        tmp_en = self.raw_en.copy()
        tmp_ch = self.raw_ch.copy()
        for i in reversed(range(len(tmp_en))):
            if len(tmp_en[i]) > 50:
                self.raw_en.pop(i)
                self.raw_ch.pop(i)

    # 读取中英文语料
    def read_raw_data(self):
        self.raw_en = []
        self.raw_ch = []
        with open(FLAGS.corpus[0], 'r') as f:
            for line in f.readlines():
                self.raw_en.append(line.lower().strip('\n').split())

        with open(FLAGS.corpus[1], 'r') as f:
            for line in f.readlines():
                self.raw_ch.append(line.strip('\n').split())

    # 建立中英文字典
    def create_dict(self):
        self.en_dict = dict()
        self.ch_dict = dict()

        tmp_raw_en = []
        for en in self.raw_en:
            tmp_raw_en.extend(en)
        counter = collections.Counter(tmp_raw_en).most_common(FLAGS.most_en - 2)

        self.en_dict['<PAD>'] = len(self.en_dict)
        self.en_dict['<UNK>'] = len(self.en_dict)
        for word, _ in counter:
            self.en_dict[word] = len(self.en_dict)

        self.en_reverse_dict = {v: k for k, v in self.en_dict.items()}

        tmp_raw_ch = []
        for ch in self.raw_ch:
            tmp_raw_ch.extend(ch)
        counter = collections.Counter(tmp_raw_ch).most_common(FLAGS.most_ch - 4)

        self.ch_dict['<PAD>'] = len(self.ch_dict)
        self.ch_dict['<EOS>'] = len(self.ch_dict)
        self.ch_dict['<UNK>'] = len(self.ch_dict)
        self.ch_dict['<GO>'] = len(self.ch_dict)
        for word, _ in counter:
            self.ch_dict[word] = len(self.ch_dict)

        self.ch_reverse_dict = {v: k for k, v in self.ch_dict.items()}

    # 语料向量化
    def text2id(self):
        self.entext2id = []
        self.entext2id_length = []
        self.chtext2id_target = []
        self.chtext2id_input = []
        self.chtext2id_input_length = []

        for en in self.raw_en:
            tmp = []
            for word in en:
                tmp.append(self.en_dict[word] if word in self.en_dict.keys() else self.en_dict['<UNK>'])
            self.entext2id.append(tmp)
            self.entext2id_length.append(len(tmp))

        for ch in self.raw_ch:
            tmp = []
            for word in ch:
                tmp.append(self.ch_dict[word] if word in self.ch_dict.keys() else self.ch_dict['<UNK>'])

            tmp1 = tmp.copy()
            tmp1.insert(0, self.ch_dict['<GO>'])
            self.chtext2id_input.append(tmp1)
            self.chtext2id_input_length.append(len(tmp1))

            tmp.append(self.ch_dict['<EOS>'])
            self.chtext2id_target.append(tmp)


class Seq2Seq():
    '''
    Seq2Seq模型
    go: start token
    eos: end token
    '''

    def __init__(self, go=0, eos=1, l_dict_en=1000, l_dict_ch=1000):
        self.go = go
        self.eos = eos
        self.l_dict_en = l_dict_en
        self.l_dict_ch = l_dict_ch

    # 建立seq2seq的tensorflow模型
    def build_model(self):
        with tf.name_scope('Input'):
            encoder_inputs = tf.placeholder(shape=[None, None], dtype=tf.int32, name='encoder_inputs')
            decoder_inputs = tf.placeholder(shape=[None, None], dtype=tf.int32, name='decoder_inputs')
            decoder_targets = tf.placeholder(shape=[None, None], dtype=tf.int32, name='decoder_targets')

            learning_rate = tf.placeholder(tf.float32, name='learning_rate')

            encoder_inputs_length = tf.placeholder(shape=[None], dtype=tf.int32, name='encoder_inputs_length')
            decoder_inputs_length = tf.placeholder(shape=[None], dtype=tf.int32, name='decoder_inputs_length')
            max_decoder_inputs_length = tf.reduce_max(decoder_inputs_length, name='max_decoder_input_length')
            max_encoder_inputs_length = tf.reduce_max(encoder_inputs_length, name='max_encoder_input_length')

            keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.name_scope('Embedding'):
            embedding_matrix_en = tf.Variable(tf.random_uniform([self.l_dict_en, FLAGS.embedding_en_size], -1.0, 1.0),
                                              dtype=tf.float32, name='embedding_matrix_en')
            embedding_matrix_ch = tf.Variable(tf.random_uniform([self.l_dict_ch, FLAGS.embedding_ch_size], -1.0, 1.0),
                                              dtype=tf.float32, name='embedding_matrix_ch')
            encoder_inputs_embeded = tf.nn.embedding_lookup(embedding_matrix_en, encoder_inputs)
            decoder_inputs_embeded = tf.nn.embedding_lookup(embedding_matrix_ch, decoder_inputs)

        with tf.variable_scope('encoder'):
            encoder_cell = self._create_rnn_cell(FLAGS.encoder_hidden_units, keep_prob)

            encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                                     inputs=encoder_inputs_embeded,
                                                                     sequence_length=encoder_inputs_length,
                                                                     dtype=tf.float32)

        with tf.variable_scope('decoder'):
            decoder_cell = self._create_rnn_cell(FLAGS.decoder_hidden_units, keep_prob)

            W_attn = tf.get_variable('W_attention', [FLAGS.decoder_hidden_units, FLAGS.encoder_hidden_units])

            output_layer = tf.layers.Dense(self.l_dict_ch, name='projection_layer')

            with tf.variable_scope('decoder'):
                training_decoder_outputs, _ = tf.nn.dynamic_rnn(cell=decoder_cell,
                                                                inputs=decoder_inputs_embeded,
                                                                sequence_length=decoder_inputs_length,
                                                                initial_state=encoder_final_state,
                                                                dtype=tf.float32)

                ei = tf.reshape(
                    tf.matmul(tf.reshape(training_decoder_outputs, [-1, FLAGS.decoder_hidden_units]), W_attn),
                    [-1, max_decoder_inputs_length, FLAGS.encoder_hidden_units])
                eij = tf.nn.softmax(tf.matmul(ei, tf.transpose(encoder_outputs, [0, 2, 1])), axis=2)
                context = tf.matmul(eij, encoder_outputs)

                training_decoder_outputs = tf.reshape(output_layer(
                    tf.reshape(tf.concat([context, training_decoder_outputs], axis=-1),
                               [-1, FLAGS.encoder_hidden_units + FLAGS.decoder_hidden_units])),
                    [-1, max_decoder_inputs_length, self.l_dict_ch])

            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
                prediction = tf.zeros([0], dtype=tf.int32)

                decoder_initial_state = encoder_final_state

                decoder_input = self.go * tf.ones_like(encoder_inputs_length, tf.int32)
                i = tf.constant(0, tf.int32)

                def cond_time_pre(i, prediction, decoder_initial_state, decoder_input):
                    return tf.less(i, max_decoder_inputs_length)

                def body_time_pre(i, prediction, decoder_initial_state, decoder_input):
                    prediction_decoder_output, decoder_initial_state = decoder_cell(
                        tf.nn.embedding_lookup(embedding_matrix_ch, decoder_input), decoder_initial_state)
                    ei = tf.matmul(prediction_decoder_output, W_attn)
                    eij = tf.nn.softmax(tf.matmul(tf.reshape(ei, [-1, 1, FLAGS.encoder_hidden_units]),
                                                  tf.transpose(encoder_outputs, (0, 2, 1))), -1)

                    context = tf.squeeze(tf.matmul(eij, encoder_outputs), 1)
                    prediction_output = output_layer(tf.concat([context, prediction_decoder_output], -1))

                    decoder_input = tf.argmax(prediction_output, 1, output_type=tf.int32)
                    prediction = tf.concat([prediction, decoder_input], axis=0)

                    return tf.add(i, 1), prediction, decoder_initial_state, decoder_input

                i, prediction, _, _ = tf.while_loop(cond_time_pre, body_time_pre,
                                                    [i, prediction, decoder_initial_state,
                                                     decoder_input],
                                                    shape_invariants=[i.get_shape(),
                                                                      tf.TensorShape([None]),
                                                                      tf.TensorShape([None,
                                                                                      2 * FLAGS.decoder_hidden_units]),
                                                                      tf.TensorShape([None])]
                                                    )
                prediction = tf.transpose(tf.reshape(prediction, [max_decoder_inputs_length, -1]))

        with tf.name_scope('Loss'):
            prediction = tf.identity(prediction, name='prediction')

            training_logits = tf.identity(training_decoder_outputs, name='logits')

            masks = tf.sequence_mask(decoder_inputs_length,
                                     max_decoder_inputs_length,
                                     dtype=tf.float32,
                                     name='masks')

            loss = tf.identity(
                seq2seq.sequence_loss(training_logits, decoder_targets, masks), name='loss')

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer')
            gradients = optimizer.compute_gradients(loss)
            clipped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(clipped_gradients, name='train_op')

        if FLAGS.graph_write:
            writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph(), filename_suffix='nmt3')
            writer.flush()
            writer.close()
            print('Graph saved successfully!')

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=1)
        saver.save(sess, FLAGS.model_save_path + 'nmt3')
        sess.close()
        print('Model saved successfully!')

    def _create_rnn_cell(self, rnn_size, keep_prob):
        def single_rnn_cell(rnn_size, keep_prob):
            # 创建单个cell，这里需要注意的是一定要使用一个single_rnn_cell的函数，不然直接把cell放在MultiRNNCell
            # 的列表中最终模型会发生错误
            single_cell = tf.nn.rnn_cell.GRUCell(rnn_size)
            # 添加dropout
            cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, output_keep_prob=keep_prob)
            return cell

        # 列表中每个元素都是调用single_rnn_cell函数
        cell = tf.nn.rnn_cell.MultiRNNCell([single_rnn_cell(rnn_size, keep_prob) for _ in range(2)],
                                           state_is_tuple=False)
        return cell

    def train(self, entext2id, entext2id_length, en_dict, chtext2id_input, chtext2id_input_length, ch_dict,
              ch_reverse_dict, chtext2id_target):
        sess = tf.Session()

        new_saver = tf.train.import_meta_graph(FLAGS.model_save_path + 'nmt3.meta')
        new_saver.restore(sess, FLAGS.model_save_path + 'nmt6_12')

        graph = tf.get_default_graph()

        encoder_inputs = graph.get_operation_by_name('Input/encoder_inputs').outputs[0]
        decoder_inputs = graph.get_operation_by_name('Input/decoder_inputs').outputs[0]
        decoder_targets = graph.get_operation_by_name('Input/decoder_targets').outputs[0]
        learning_rate = graph.get_operation_by_name('Input/learning_rate').outputs[0]
        keep_prob = graph.get_operation_by_name('Input/keep_prob').outputs[0]

        encoder_inputs_length = graph.get_operation_by_name('Input/encoder_inputs_length').outputs[0]
        decoder_inputs_length = graph.get_operation_by_name('Input/decoder_inputs_length').outputs[0]

        prediction = graph.get_tensor_by_name('Loss/prediction:0')
        loss = graph.get_tensor_by_name('Loss/loss:0')
        train_op = graph.get_operation_by_name('Loss/train_op')

        test = ['In old China , the people did not have human rights .',
                'The physical conditions of women have greatly improved .',
                'In old china there was not even the most basic medical and health service for ordinary people .',
                'China has cracked down on serious criminal offences in accordance with law .',
                'Children are the future and hope of mankind .']
        test_strip = [test[i].lower().split() for i in range(len(test))]
        test_len = [len(test_strip[i]) for i in range(len(test))]
        print(test_len)
        test2id = []
        for i in range(len(test)):
            tmp = []
            for word in test_strip[i]:
                tmp.append(en_dict[word] if word in en_dict.keys() else en_dict['<UNK>'])
            test2id.append(tmp)
        max_test_len = np.max(test_len)
        for i in range(len(test_len)):
            if test_len[i] < max_test_len:
                test2id[i] += [en_dict['<PAD>']] * (max_test_len - test_len[i])

        test_encoder_input = test2id
        test_encoder_input_length = test_len
        test_decoder_inputs_length = [15, 20, 30, 20, 10]

        print(test_encoder_input)

        m_samples = len(entext2id)
        total_batch = m_samples // FLAGS.batch_size

        loss_ = []
        for epoch in range(1, FLAGS.epochs + 1):
            loss_epoch = 0.0
            for batch in range(total_batch):
                prediction_ = sess.run(prediction, feed_dict={encoder_inputs: test_encoder_input,
                                                              encoder_inputs_length: test_encoder_input_length,
                                                              decoder_inputs_length: test_decoder_inputs_length,
                                                              keep_prob: 1.0})

                sys.stdout.write('>> %d/%d | %d/%d\n' % (epoch, FLAGS.epochs, batch + 1, total_batch))

                for i_test in range(len(test)):
                    tmp = []
                    for idx in prediction_[i_test]:
                        if idx == ch_dict['<EOS>']:
                            break
                        tmp.append(ch_reverse_dict[idx])
                    sys.stdout.write('English: %s\n' % (test[i_test]))
                    sys.stdout.write('Chinese: %s\n\n' % (''.join(tmp)))
                sys.stdout.write(
                    '-------------------------------------------------------------------------------------------------\n')
                sys.stdout.flush()

                x_input_batch = entext2id[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size]
                y_input_batch = chtext2id_input[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size]
                y_target_batch = chtext2id_target[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size]

                x_input_batch_length = entext2id_length[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size]
                y_input_batch_length = chtext2id_input_length[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size]

                x_input_batch = self.padding(x_input_batch, x_input_batch_length, en_dict['<PAD>'])
                y_input_batch = self.padding(y_input_batch, y_input_batch_length, ch_dict['<PAD>'])
                y_target_batch = self.padding(y_target_batch, y_input_batch_length, ch_dict['<PAD>'])

                feed_dict = {
                    encoder_inputs: x_input_batch,
                    decoder_inputs: y_input_batch,
                    decoder_targets: y_target_batch,
                    learning_rate: FLAGS.lr,
                    encoder_inputs_length: x_input_batch_length,
                    decoder_inputs_length: y_input_batch_length,
                    keep_prob: FLAGS.keep_prob
                }
                _, loss_batch = sess.run([train_op, loss], feed_dict=feed_dict)

                loss_epoch += loss_batch
            loss_.append(loss_epoch / total_batch)

            print('\033[1;31;40m')
            print('>> %d/%d | Loss:%.9f' % (epoch, FLAGS.epochs, loss_[-1]))
            print('\033[0m')

            r = np.random.permutation(m_samples)
            entext2id = self.rearrange(entext2id, r)
            chtext2id_input = self.rearrange(chtext2id_input, r)
            chtext2id_target = self.rearrange(chtext2id_target, r)
            entext2id_length = self.rearrange(entext2id_length, r)
            chtext2id_input_length = self.rearrange(chtext2id_input_length, r)

            if epoch % FLAGS.per_save == 0:
                new_saver.save(sess, FLAGS.model_save_path + 'nmt3')
                print('Model saved successfully!')

        fig = plt.figure(figsize=(10, 8))
        plt.plot(loss_)
        plt.savefig(FLAGS.model_save_path + 'nmt3_loss.png', bbox_inches='tight')
        plt.close(fig)

    def predict(self, en_dict, ch_dict, ch_reverse_dict):
        sess = tf.Session()
        tf.train.import_meta_graph(FLAGS.model_save_path + 'nmt3.meta').restore(sess,
                                                                                   FLAGS.model_save_path + 'nmt3')
        graph = tf.get_default_graph()

        encoder_inputs = graph.get_operation_by_name('Input/encoder_inputs').outputs[0]
        keep_prob = graph.get_operation_by_name('Input/keep_prob').outputs[0]
        encoder_inputs_length = graph.get_operation_by_name('Input/encoder_inputs_length').outputs[0]
        decoder_inputs_length = graph.get_operation_by_name('Input/decoder_inputs_length').outputs[0]
        prediction = graph.get_tensor_by_name('Loss/prediction:0')

        print('\033[1;31;40m')
        while 1:
            test = input('Please enter English sentence or q to quit\n')
            if test.lower() == 'q':
                break
            test_strip = [WordPunctTokenizer().tokenize(test.lower().strip(' '))]
            test_len = [len(test_strip[0])]
            test2id = []
            tmp = []
            for word in test_strip[0]:
                tmp.append(en_dict[word] if word in en_dict.keys() else en_dict['<UNK>'])
            test2id.append(tmp)

            test_encoder_input = test2id
            test_encoder_input_length = test_len
            test_decoder_inputs_length = [30]

            prediction_ = sess.run(prediction, feed_dict={encoder_inputs: test_encoder_input,
                                                          encoder_inputs_length: test_encoder_input_length,
                                                          decoder_inputs_length: test_decoder_inputs_length,
                                                          keep_prob: 1.0})

            tmp = []
            for idx in prediction_[0]:
                if idx == ch_dict['<EOS>']:
                    break
                tmp.append(ch_reverse_dict[idx])
            sys.stdout.write('\n')
            sys.stdout.write('English: %s\n' % (test))
            sys.stdout.write('Chinese: %s\n\n' % (''.join(tmp)))
            sys.stdout.flush()

        print('\033[0m')

    def padding(self, x, l, padding_id):
        l_max = np.max(l)
        return [x[i] + [padding_id] * (l_max - l[i]) for i in range(len(x))]

    def rearrange(self, x, r):
        return [x[ri] for ri in r]


def main(unused_argv):
    lang = Lang()
    print(len(lang.raw_en), lang.raw_en[:10])
    print(len(lang.raw_ch), lang.raw_ch[:10])

    print(len(lang.en_dict), [lang.en_reverse_dict[i] for i in range(100)])
    print(len(lang.ch_dict), [lang.ch_reverse_dict[i] for i in range(100)])

    print(lang.entext2id_length[:10], np.max(lang.entext2id_length), np.min(lang.entext2id_length),
          np.argmin(lang.entext2id_length), lang.entext2id[:10])
    print(lang.chtext2id_input_length[:10], np.max(lang.chtext2id_input_length), np.min(lang.chtext2id_input_length),
          np.argmin(lang.chtext2id_input_length), lang.chtext2id_target[:10])
    print(lang.chtext2id_input_length[:10], np.max(lang.chtext2id_input_length), np.min(lang.chtext2id_input_length),
          np.argmin(lang.chtext2id_input_length), lang.chtext2id_input[:10])


    if FLAGS.mode.startswith('train'):
        f = open(sys.path[0] + '/bps2.txt', 'w+')
        for i in range(len(lang.entext2id)):
            f.write(' '.join(lang.raw_ch[i]) + '\n')
            f.write(' '.join(lang.raw_en[i]) + '\n')
            f.write('\n')
        f.close()

    myseq2seq = Seq2Seq(go=lang.ch_dict['<GO>'], eos=lang.ch_dict['<EOS>'],
                        l_dict_en=len(lang.en_dict), l_dict_ch=len(lang.ch_dict)
                        )

    if FLAGS.mode == 'train0':  # train first time or retrain
        if not os.path.exists(FLAGS.model_save_path):
            os.makedirs(FLAGS.model_save_path)

        myseq2seq.build_model()

        myseq2seq.train(lang.entext2id, lang.entext2id_length, lang.en_dict,
                        lang.chtext2id_input, lang.chtext2id_input_length,
                        lang.ch_dict, lang.ch_reverse_dict,
                        lang.chtext2id_target
                        )
    elif FLAGS.mode == 'train1':  # continue train
        myseq2seq.train(lang.entext2id, lang.entext2id_length, lang.en_dict,
                        lang.chtext2id_input, lang.chtext2id_input_length,
                        lang.ch_dict, lang.ch_reverse_dict,
                        lang.chtext2id_target
                        )
    elif FLAGS.mode == 'predict':
        myseq2seq.predict(lang.en_dict, lang.ch_dict, lang.ch_reverse_dict)


if __name__ == '__main__':
    tf.app.run()
