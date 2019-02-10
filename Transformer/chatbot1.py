'''
    Transformer model for chatbot
'''
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
import sys
import pickle
import os
import jieba

tf.flags.DEFINE_integer('maxword', 50, 'max length of any sentences')
tf.flags.DEFINE_integer('block', 2, 'number of Encoder submodel')
tf.flags.DEFINE_integer('head', 8, 'number of multi_head attention')

tf.flags.DEFINE_string('model_save_path', 'model/', 'The path where model shall be saved')
tf.flags.DEFINE_integer('batch_size', 128, 'Batch size during training')
tf.flags.DEFINE_integer('epochs', 50000, 'Epochs during training')
tf.flags.DEFINE_float('lr', 0.0001, 'Initial learing rate')
tf.flags.DEFINE_integer('embedding_qa_size', 512, 'Embedding size for english words')
tf.flags.DEFINE_boolean('graph_write', True, 'whether the compute graph is written to logs file')
tf.flags.DEFINE_float('keep_prob', 0.5, 'The probility used to dropout')
tf.flags.DEFINE_string('mode', 'train0', 'The mode of train or predict as follows: '
                                         'train0: train first time or retrain'
                                         'train1: continue train'
                                         'predict: predict')
tf.flags.DEFINE_integer('per_save', 10, 'save model for every per_save')

FLAGS = tf.flags.FLAGS


def layer_norm(x, scale, bias, epsilon=1.0e-8):
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


class Transformer():
    '''
    Transformer模型
    go: start token
    eos: end token
    l_dict_qa: number of word in english dictionary
    config: parameters for shell
    '''

    def __init__(self, go=0, eos=1, l_dict_qa=1000, config=FLAGS):
        self.go = go
        self.eos = eos
        self.l_dict_qa = l_dict_qa
        self.config = config

    # 建立seq2seq的tensorflow模型
    def build_model(self):
        with tf.name_scope('Input'):
            encoder_inputs = tf.placeholder(shape=[None, None], dtype=tf.int32, name='encoder_inputs')
            decoder_inputs = tf.placeholder(shape=[None, None], dtype=tf.int32, name='decoder_inputs')
            decoder_targets = tf.placeholder(shape=[None, None], dtype=tf.int32, name='decoder_targets')

            encoder_length = tf.placeholder(shape=[None], dtype=tf.int32, name='encoder_length')
            decoder_length = tf.placeholder(shape=[None], dtype=tf.int32, name='decoder_length')

            encoder_pos = tf.placeholder(tf.float32, [None, self.config.embedding_qa_size], name='encoder_position')
            decoder_pos = tf.placeholder(tf.float32, [None, self.config.embedding_qa_size], name='decoder_position')

            keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            max_encoder_length = tf.reduce_max(encoder_length)
            max_decoder_length = tf.reduce_max(decoder_length)
            padding_mask_encoder = tf.tile(tf.expand_dims(tf.greater(encoder_inputs, 0), 1),
                                           [self.config.head, tf.shape(encoder_inputs)[1], 1])
            padding_mask_decoder = tf.tile(tf.expand_dims(tf.greater(decoder_inputs, 0), 1),
                                           [self.config.head, tf.shape(decoder_inputs)[1], 1])
            padding_mask_decoder_encoder = tf.tile(tf.expand_dims(tf.greater(encoder_inputs, 0), 1),
                                                   [self.config.head, tf.shape(decoder_inputs)[1], 1])

            sequence_mask_decoder = tf.sequence_mask(decoder_length, max_decoder_length)

        with tf.name_scope('Embedding'):
            embedding_matrix = tf.Variable(
                tf.random_uniform([self.l_dict_qa, self.config.embedding_qa_size], -1.0, 1.0),
                dtype=tf.float32, name='embedding_matrix_ch')
            encoder_embeded = tf.nn.embedding_lookup(embedding_matrix, encoder_inputs)
            decoder_embeded = tf.nn.embedding_lookup(embedding_matrix, decoder_inputs)

        with tf.variable_scope('encoder'):
            encoder_p = tf.nn.dropout(encoder_embeded + encoder_pos[:max_encoder_length], keep_prob)

            for block in range(self.config.block):
                with tf.variable_scope('selfattention' + str(block)):
                    WQ = tf.layers.Dense(self.config.embedding_qa_size, use_bias=False, name='Q')
                    WK = tf.layers.Dense(self.config.embedding_qa_size, use_bias=False, name='K')
                    WV = tf.layers.Dense(self.config.embedding_qa_size, use_bias=False, name='V')
                    WO = tf.layers.Dense(self.config.embedding_qa_size, use_bias=False, name='O')

                    Q = tf.concat(tf.split(WQ(encoder_p), self.config.head, axis=-1), axis=0)
                    K = tf.concat(tf.split(WK(encoder_p), self.config.head, axis=-1), axis=0)
                    V = tf.concat(tf.split(WV(encoder_p), self.config.head, axis=-1), axis=0)

                    QK = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(
                        self.config.embedding_qa_size / self.config.head)
                    Z_p = WO(tf.concat(
                        tf.split(tf.matmul(softmax(QK, padding_mask_encoder), V),
                                 self.config.head, axis=0), axis=-1))

                    scale_sa = tf.get_variable('scale_sa',
                                               initializer=tf.ones([self.config.embedding_qa_size],
                                                                   dtype=tf.float32))
                    bias_sa = tf.get_variable('bias_sa',
                                              initializer=tf.zeros([self.config.embedding_qa_size],
                                                                   dtype=tf.float32))

                    encoder_p = tf.nn.dropout(layer_norm(encoder_p + Z_p, scale_sa, bias_sa), keep_prob)

                with tf.variable_scope('feedforward' + str(block)):
                    ffrelu = tf.layers.Dense(4 * self.config.embedding_qa_size, activation=tf.nn.relu, name='ffrelu')
                    ff = tf.layers.Dense(self.config.embedding_qa_size, name='ff')
                    scale_ff = tf.get_variable('scale_ff',
                                               initializer=tf.ones([self.config.embedding_qa_size],
                                                                   dtype=tf.float32))
                    bias_ff = tf.get_variable('bias_ff',
                                              initializer=tf.zeros([self.config.embedding_qa_size],
                                                                   dtype=tf.float32))

                    encoder_p = tf.nn.dropout(layer_norm(ff(ffrelu(encoder_p)) + encoder_p, scale_ff, bias_ff),
                                              keep_prob)

        with tf.variable_scope('decoder'):
            # for train
            with tf.variable_scope('decoder'):
                decoder_p = tf.nn.dropout(decoder_embeded + decoder_pos[:max_decoder_length], keep_prob)

                future_mask = tf.tile(
                    tf.expand_dims(tf.sequence_mask(tf.range(1, limit=tf.shape(decoder_inputs)[1] + 1)), 0),
                    [tf.shape(decoder_inputs)[0] * self.config.head, 1, 1])
                future_mask_final = padding_mask_decoder & future_mask

                for block in range(self.config.block):
                    with tf.variable_scope('mask_selfattention' + str(block)):
                        WQ = tf.layers.Dense(self.config.embedding_qa_size, use_bias=False, name='Q')
                        WK = tf.layers.Dense(self.config.embedding_qa_size, use_bias=False, name='K')
                        WV = tf.layers.Dense(self.config.embedding_qa_size, use_bias=False, name='V')
                        WO = tf.layers.Dense(self.config.embedding_qa_size, use_bias=False, name='O')

                        Q = tf.concat(tf.split(WQ(decoder_p), self.config.head, axis=-1), axis=0)
                        K = tf.concat(tf.split(WK(decoder_p), self.config.head, axis=-1), axis=0)
                        V = tf.concat(tf.split(WV(decoder_p), self.config.head, axis=-1), axis=0)

                        QK = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(
                            self.config.embedding_qa_size / self.config.head)
                        Z_p = WO(tf.concat(
                            tf.split(tf.matmul(softmax(QK, future_mask_final), V),
                                     self.config.head, axis=0), axis=-1))

                        scale_msa = tf.get_variable('scale_msa', initializer=tf.ones([self.config.embedding_qa_size],
                                                                                     dtype=tf.float32))
                        bias_msa = tf.get_variable('bias_msa', initializer=tf.zeros([self.config.embedding_qa_size],
                                                                                    dtype=tf.float32))

                        decoder_p = tf.nn.dropout(layer_norm(decoder_p + Z_p, scale_msa, bias_msa), keep_prob)

                    with tf.variable_scope('encoder_decoder_attention' + str(block)):
                        WQ = tf.layers.Dense(self.config.embedding_qa_size, use_bias=False, name='Q')
                        WK = tf.layers.Dense(self.config.embedding_qa_size, use_bias=False, name='K')
                        WV = tf.layers.Dense(self.config.embedding_qa_size, use_bias=False, name='V')
                        WO = tf.layers.Dense(self.config.embedding_qa_size, use_bias=False, name='O')

                        Q = tf.concat(tf.split(WQ(decoder_p), self.config.head, axis=-1), axis=0)
                        K = tf.concat(tf.split(WK(encoder_p), self.config.head, axis=-1), axis=0)
                        V = tf.concat(tf.split(WV(encoder_p), self.config.head, axis=-1), axis=0)

                        QK = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(
                            self.config.embedding_qa_size / self.config.head)
                        Z_p = WO(tf.concat(
                            tf.split(
                                tf.matmul(softmax(QK, padding_mask_decoder_encoder), V),
                                self.config.head, axis=0), axis=-1))

                        scale_eda = tf.get_variable('scale_eda', initializer=tf.ones([self.config.embedding_qa_size],
                                                                                     dtype=tf.float32))
                        bias_eda = tf.get_variable('bias_eda', initializer=tf.zeros([self.config.embedding_qa_size],
                                                                                    dtype=tf.float32))

                        decoder_p = tf.nn.dropout(layer_norm(decoder_p + Z_p, scale_eda, bias_eda), keep_prob)

                    with tf.variable_scope('feedforward' + str(block)):
                        ffrelu = tf.layers.Dense(4 * self.config.embedding_qa_size, activation=tf.nn.relu,
                                                 name='ffrelu')
                        ff = tf.layers.Dense(self.config.embedding_qa_size, name='ff')
                        scale_ff = tf.get_variable('scale_ff', initializer=tf.ones([self.config.embedding_qa_size],
                                                                                   dtype=tf.float32))
                        bias_ff = tf.get_variable('bias_ff', initializer=tf.zeros([self.config.embedding_qa_size],
                                                                                  dtype=tf.float32))

                        decoder_p = tf.nn.dropout(layer_norm(ff(ffrelu(decoder_p)) + decoder_p, scale_ff, bias_ff),
                                                  keep_prob)
                output_layer = tf.layers.Dense(self.l_dict_qa, name='project')
                logits = output_layer(decoder_p)

            # for inference
            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
                k = tf.constant(0, dtype=tf.int32)
                decoder_infer_inputs = self.go * tf.ones([tf.shape(decoder_length)[0], k + 1], dtype=tf.int32)
                decoder_infer_outputs = tf.zeros([tf.shape(decoder_length)[0], 0], dtype=tf.int32)

                def cond(k, decoder_infer_inputs, decoder_infer_outputs):
                    return tf.less(k, max_decoder_length)

                def body(k, decoder_infer_inputs, decoder_infer_outputs):
                    decoder_infer = tf.concat([tf.nn.embedding_lookup(embedding_matrix, decoder_infer_inputs),
                                               tf.tile(tf.expand_dims(decoder_pos[:(k + 1)], 0),
                                                       [tf.shape(decoder_length)[0], 1, 1])], axis=-1)
                    padding_mask_decoder_infer = tf.tile(tf.expand_dims(tf.greater(decoder_infer_inputs, 0), 1),
                                                         [self.config.head, tf.shape(decoder_infer_inputs)[1], 1])
                    future_mask_infer = tf.tile(
                        tf.expand_dims(tf.sequence_mask(tf.range(1, limit=tf.shape(decoder_infer_inputs)[1] + 1)), 0),
                        [tf.shape(decoder_infer_inputs)[0] * self.config.head, 1, 1])
                    future_mask_final_infer = padding_mask_decoder_infer & future_mask_infer
                    padding_mask_decoder_encoder_infer = tf.tile(tf.expand_dims(tf.greater(encoder_inputs, 0), 1),
                                                                 [self.config.head, tf.shape(decoder_infer_inputs)[1],
                                                                  1],
                                                                 name='mask_decoder_encoder')

                    for block in range(self.config.block):
                        with tf.variable_scope('mask_selfattention' + str(block)):
                            WQ = tf.layers.Dense(self.config.embedding_qa_size, use_bias=False, name='Q')
                            WK = tf.layers.Dense(self.config.embedding_qa_size, use_bias=False, name='K')
                            WV = tf.layers.Dense(self.config.embedding_qa_size, use_bias=False, name='V')
                            WO = tf.layers.Dense(self.config.embedding_qa_size, use_bias=False, name='O')

                            Q = tf.concat(tf.split(WQ(decoder_infer), self.config.head, axis=-1), axis=0)
                            K = tf.concat(tf.split(WK(decoder_infer), self.config.head, axis=-1), axis=0)
                            V = tf.concat(tf.split(WV(decoder_infer), self.config.head, axis=-1), axis=0)

                            QK = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(
                                self.config.embedding_qa_size / self.config.head)
                            Z_p = WO(tf.concat(
                                tf.split(
                                    tf.matmul(softmax(QK, future_mask_final_infer), V),
                                    self.config.head, axis=0), axis=-1))

                            scale_msa = tf.get_variable('scale_msa',
                                                        initializer=tf.ones([self.config.embedding_qa_size],
                                                                            dtype=tf.float32))
                            bias_msa = tf.get_variable('bias_msa',
                                                       initializer=tf.zeros([self.config.embedding_qa_size],
                                                                            dtype=tf.float32))

                            decoder_infer = layer_norm(decoder_infer + Z_p, scale_msa, bias_msa)

                        with tf.variable_scope('encoder_decoder_attention' + str(block)):
                            WQ = tf.layers.Dense(self.config.embedding_qa_size, use_bias=False, name='Q')
                            WK = tf.layers.Dense(self.config.embedding_qa_size, use_bias=False, name='K')
                            WV = tf.layers.Dense(self.config.embedding_qa_size, use_bias=False, name='V')
                            WO = tf.layers.Dense(self.config.embedding_qa_size, use_bias=False, name='O')

                            Q = tf.concat(tf.split(WQ(decoder_infer), self.config.head, axis=-1), axis=0)
                            K = tf.concat(tf.split(WK(encoder_p), self.config.head, axis=-1), axis=0)
                            V = tf.concat(tf.split(WV(encoder_p), self.config.head, axis=-1), axis=0)

                            QK = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(
                                self.config.embedding_qa_size / self.config.head)
                            Z_p = WO(tf.concat(tf.split(
                                tf.matmul(softmax(QK, padding_mask_decoder_encoder_infer), V),
                                self.config.head, axis=0), axis=-1))

                            scale_eda = tf.get_variable('scale_eda',
                                                        initializer=tf.ones([self.config.embedding_qa_size],
                                                                            dtype=tf.float32))
                            bias_eda = tf.get_variable('bias_eda',
                                                       initializer=tf.zeros([self.config.embedding_qa_size],
                                                                            dtype=tf.float32))

                            decoder_infer = layer_norm(decoder_infer + Z_p, scale_eda, bias_eda)

                        with tf.variable_scope('feedforward' + str(block)):
                            ffrelu = tf.layers.Dense(4 * self.config.embedding_qa_size, activation=tf.nn.relu,
                                                     name='ffrelu')
                            ff = tf.layers.Dense(self.config.embedding_qa_size, name='ff')
                            scale_ff = tf.get_variable('scale_ff',
                                                       initializer=tf.ones([self.config.embedding_qa_size],
                                                                           dtype=tf.float32))
                            bias_ff = tf.get_variable('bias_ff',
                                                      initializer=tf.zeros([self.config.embedding_qa_size],
                                                                           dtype=tf.float32))

                            decoder_infer = layer_norm(ff(ffrelu(decoder_infer)) + decoder_infer, scale_ff, bias_ff)

                    output_layer = tf.layers.Dense(self.l_dict_qa, name='project')
                    infer_logits = output_layer(decoder_infer)

                    decoder_infer_outputs_tmp = tf.argmax(infer_logits[:, -1:], axis=-1, output_type=tf.int32)
                    decoder_infer_outputs_tmpf = tf.zeros_like(decoder_infer_outputs_tmp)
                    decoder_infer_outputs = tf.concat(
                        [decoder_infer_outputs, tf.where(sequence_mask_decoder[:, k:(k + 1)], decoder_infer_outputs_tmp,
                                                         decoder_infer_outputs_tmpf)], axis=-1)

                    decoder_infer_inputs_tmp = decoder_infer_outputs[:, -1:]
                    decoder_infer_inputs_tmpf = tf.zeros_like(decoder_infer_inputs_tmp)
                    eos_mask = tf.not_equal(decoder_infer_inputs_tmp, self.eos * tf.ones_like(decoder_infer_inputs_tmp))

                    decoder_infer_inputs = tf.concat(
                        [decoder_infer_inputs, tf.where(eos_mask, decoder_infer_inputs_tmp, decoder_infer_inputs_tmpf)],
                        axis=-1)

                    return tf.add(k, 1), decoder_infer_inputs, decoder_infer_outputs

                _, _, decoder_infer_outputs = tf.while_loop(cond, body,
                                                            [k, decoder_infer_inputs, decoder_infer_outputs],
                                                            shape_invariants=[k.get_shape(),
                                                                              tf.TensorShape(
                                                                                  [decoder_length.get_shape().as_list()[
                                                                                       0], None]),
                                                                              tf.TensorShape(
                                                                                  [decoder_length.get_shape().as_list()[
                                                                                       0], None])
                                                                              ])
        with tf.name_scope('Loss'):
            prediction = tf.argmax(logits, axis=-1, output_type=tf.int32)
            predictionf = tf.zeros_like(prediction)
            prediction = tf.where(sequence_mask_decoder, prediction, predictionf)
            prediction = tf.identity(prediction, name='prediction')

            accuracy = tf.cast(tf.equal(prediction, decoder_targets), tf.float32)
            accuracyf = tf.zeros_like(accuracy)
            accuracy = tf.where(sequence_mask_decoder, accuracy, accuracyf)
            accuracy = tf.reduce_sum(accuracy) / tf.cast(tf.reduce_sum(decoder_length), tf.float32)
            accuracy = tf.identity(accuracy, name='accuracy')

            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_targets, logits=logits)
            costf = tf.zeros_like(cost)

            loss = tf.reduce_mean(tf.div(tf.reduce_sum(tf.where(sequence_mask_decoder, cost, costf), axis=-1),
                                         tf.cast(decoder_length, tf.float32)), name='loss')

            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
            train_op = optimizer.minimize(loss, name='train_op')

            prediction_infer = tf.identity(decoder_infer_outputs, name='prediction_infer')

        if FLAGS.graph_write:
            writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph(), filename_suffix='chatbot2')
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

        print('Total number of parameters: %d' % number_trainable_variables)

        saver = tf.train.Saver(max_to_keep=1)
        saver.save(sess, FLAGS.model_save_path + 'chatbot2')
        sess.close()
        print('Model saved successfully!')

    def train(self, qtext2id, qtext2id_length, atext2id_input, atext2id_target, atext2id_input_length,
              qa_dict, qa_reverse_dict):
        sess = tf.Session()

        new_saver = tf.train.import_meta_graph(self.config.model_save_path + 'chatbot2.meta')
        new_saver.restore(sess, self.config.model_save_path + 'chatbot2')

        graph = tf.get_default_graph()

        encoder_inputs = graph.get_operation_by_name('Input/encoder_inputs').outputs[0]
        decoder_inputs = graph.get_operation_by_name('Input/decoder_inputs').outputs[0]
        decoder_targets = graph.get_operation_by_name('Input/decoder_targets').outputs[0]
        keep_prob = graph.get_operation_by_name('Input/keep_prob').outputs[0]

        encoder_length = graph.get_operation_by_name('Input/encoder_length').outputs[0]
        decoder_length = graph.get_operation_by_name('Input/decoder_length').outputs[0]

        encoder_pos = graph.get_operation_by_name('Input/encoder_position').outputs[0]
        decoder_pos = graph.get_operation_by_name('Input/decoder_position').outputs[0]

        loss = graph.get_tensor_by_name('Loss/loss:0')
        train_op = graph.get_operation_by_name('Loss/train_op')

        prediction = graph.get_tensor_by_name('Loss/prediction:0')
        accuracy = graph.get_tensor_by_name('Loss/accuracy:0')

        prediction_infer = graph.get_tensor_by_name('Loss/prediction_infer:0')

        test = ['你真好', '吃完饭去干什么呢？']
        test_strip = [jieba.lcut(test[i]) for i in range(len(test))]
        test_len = [len(test_strip[i]) for i in range(len(test))]
        print(test_len)
        test2id = []
        for i in range(len(test)):
            tmp = []
            for word in test_strip[i]:
                tmp.append(qa_dict[word] if word in qa_dict.keys() else qa_dict['<UNK>'])
            test2id.append(tmp)
        max_test_len = np.max(test_len)
        for i in range(len(test_len)):
            if test_len[i] < max_test_len:
                test2id[i] += [qa_dict['<PAD>']] * (max_test_len - test_len[i])

        test_encoder_input = test2id
        test_encoder_input_length = test_len
        test_decoder_inputs_length = [10, 12]

        print(test_encoder_input)

        m_samples = len(qtext2id)
        total_batch = m_samples // self.config.batch_size

        pos_encoder = np.array(
            [[position / np.power(10000.0, 2.0 * (i // 2) / self.config.embedding_qa_size) for i in
              range(self.config.embedding_qa_size)]
             for position in range(self.config.maxword)])
        pos_encoder[:, 0::2] = np.sin(pos_encoder[:, 0::2])
        pos_encoder[:, 1::2] = np.cos(pos_encoder[:, 1::2])

        pos_decoder = np.array(
            [[position / np.power(10000.0, 2.0 * (i // 2) / self.config.embedding_qa_size) for i in
              range(self.config.embedding_qa_size)]
             for position in range(self.config.maxword)])
        pos_decoder[:, 0::2] = np.sin(pos_decoder[:, 0::2])
        pos_decoder[:, 1::2] = np.cos(pos_decoder[:, 1::2])

        loss_ = []
        acc_ = []
        for epoch in range(1, self.config.epochs + 1):
            loss_epoch = 0.0
            acc_epoch = 0.0
            for batch in range(total_batch):
                x_input_batch = qtext2id[batch * self.config.batch_size:(batch + 1) * self.config.batch_size]
                y_input_batch = atext2id_input[batch * self.config.batch_size:(batch + 1) * self.config.batch_size]
                y_target_batch = atext2id_target[batch * self.config.batch_size:(batch + 1) * self.config.batch_size]

                x_input_batch_length = qtext2id_length[
                                       batch * self.config.batch_size:(batch + 1) * self.config.batch_size]
                y_input_batch_length = atext2id_input_length[
                                       batch * self.config.batch_size:(batch + 1) * self.config.batch_size]

                x_input_batch = self.padding(x_input_batch, x_input_batch_length, qa_dict['<PAD>'])
                y_input_batch = self.padding(y_input_batch, y_input_batch_length, qa_dict['<PAD>'])
                y_target_batch = self.padding(y_target_batch, y_input_batch_length, qa_dict['<PAD>'])

                feed_dict = {
                    encoder_inputs: x_input_batch,
                    decoder_inputs: y_input_batch,
                    decoder_targets: y_target_batch,
                    encoder_length: x_input_batch_length,
                    decoder_length: y_input_batch_length,
                    encoder_pos: pos_encoder,
                    decoder_pos: pos_decoder,
                    keep_prob: self.config.keep_prob
                }
                prediction_, acc_batch, loss_batch, _ = sess.run([prediction, accuracy, loss, train_op],
                                                                 feed_dict=feed_dict)
                sys.stdout.write('>> %d/%d | %d/%d  loss:%.9f   acc:%.2f%%\n' % (
                    epoch, self.config.epochs, batch + 1, total_batch, loss_batch, 100.0 * acc_batch))
                sys.stdout.flush()

                prediction_infer_ = sess.run(prediction_infer, feed_dict={encoder_inputs: test_encoder_input,
                                                                          encoder_length: test_encoder_input_length,
                                                                          decoder_length: test_decoder_inputs_length,
                                                                          encoder_pos: pos_encoder,
                                                                          decoder_pos: pos_decoder,
                                                                          keep_prob: 1.0})

                for i_test in range(len(test)):
                    tmp = []
                    for idx in prediction_infer_[i_test]:
                        if idx == qa_dict['<EOS>']:
                            break
                        tmp.append(qa_reverse_dict[idx])
                    sys.stdout.write('问: %s\n' % (test[i_test]))
                    sys.stdout.write('答: %s\n\n' % (''.join(tmp)))
                sys.stdout.write(
                    '-------------------------------------------------------------------------------------------------\n')
                sys.stdout.flush()

                loss_epoch += loss_batch
                acc_epoch += acc_batch
            loss_.append(loss_epoch / total_batch)
            acc_.append(acc_epoch / total_batch)

            print('\033[1;31;40m')
            print('>> %d/%d | Loss:%.9f Acc:%.2f%%\n' % (epoch, self.config.epochs, loss_[-1], 100. * acc_[-1]))
            print('\033[0m')

            r = np.random.permutation(m_samples)
            qtext2id = self.rearrange(qtext2id, r)
            atext2id_input = self.rearrange(atext2id_input, r)
            atext2id_target = self.rearrange(atext2id_target, r)
            qtext2id_length = self.rearrange(qtext2id_length, r)
            atext2id_input_length = self.rearrange(atext2id_input_length, r)

            if epoch % self.config.per_save == 0:
                new_saver.save(sess, self.config.model_save_path + 'chatbot2')
                print('Model saved successfully!')

        fig = plt.figure(figsize=(10, 8))
        plt.plot(loss_)
        plt.savefig(self.config.model_save_path + 'Transformer_Loss.png', bbox_inches='tight')
        plt.close(fig)

    def predict(self, qa_dict, qa_reverse_dict):
        sess = tf.Session()
        tf.train.import_meta_graph(self.config.model_save_path + 'chatbot2.meta').restore(sess,
                                                                                          self.config.model_save_path + 'chatbot2')
        graph = tf.get_default_graph()

        encoder_inputs = graph.get_operation_by_name('Input/encoder_inputs').outputs[0]
        keep_prob = graph.get_operation_by_name('Input/keep_prob').outputs[0]
        encoder_length = graph.get_operation_by_name('Input/encoder_length').outputs[0]
        decoder_length = graph.get_operation_by_name('Input/decoder_length').outputs[0]
        encoder_pos = graph.get_operation_by_name('Input/encoder_position').outputs[0]
        decoder_pos = graph.get_operation_by_name('Input/decoder_position').outputs[0]

        prediction_infer = graph.get_tensor_by_name('Loss/prediction_infer:0')

        pos_encoder = np.array(
            [[position / np.power(10000.0, 2.0 * (i // 2) / self.config.embedding_qa_size) for i in
              range(self.config.embedding_qa_size)]
             for position in range(self.config.maxword)])
        pos_encoder[:, 0::2] = np.sin(pos_encoder[:, 0::2])
        pos_encoder[:, 1::2] = np.cos(pos_encoder[:, 1::2])

        pos_decoder = np.array(
            [[position / np.power(10000.0, 2.0 * (i // 2) / self.config.embedding_qa_size) for i in
              range(self.config.embedding_qa_size)]
             for position in range(self.config.maxword)])
        pos_decoder[:, 0::2] = np.sin(pos_decoder[:, 0::2])
        pos_decoder[:, 1::2] = np.cos(pos_decoder[:, 1::2])

        print('\033[1;31;40m')

        while 1:
            test = input('请输入enter继续或者q退出\n')
            if test.lower() == 'q':
                break

            test = input('问: ')
            test_strip = [jieba.lcut(test)]
            test_len = [len(test_strip[0])]
            test2id = []
            tmp = []
            for word in test_strip[0]:
                tmp.append(qa_dict[word] if word in qa_dict.keys() else qa_dict['<UNK>'])
            test2id.append(tmp)

            test_encoder_input = test2id
            test_encoder_length = test_len
            test_decoder_length = [30]

            prediction_infer_ = sess.run(prediction_infer, feed_dict={encoder_inputs: test_encoder_input,
                                                                      encoder_length: test_encoder_length,
                                                                      decoder_length: test_decoder_length,
                                                                      encoder_pos: pos_encoder,
                                                                      decoder_pos: pos_decoder,
                                                                      keep_prob: 1.0})

            tmp = []
            for idx in prediction_infer_[0]:
                if idx == qa_dict['<EOS>']:
                    break
                tmp.append(qa_reverse_dict[idx])
            sys.stdout.write('\n')
            sys.stdout.write('问: %s\n' % (test))
            sys.stdout.write('答: %s\n\n' % (''.join(tmp)))
            sys.stdout.flush()

        print('\033[0m')

    def padding(self, x, l, padding_id):
        l_max = np.max(l)
        return [x[i] + [padding_id] * (l_max - l[i]) for i in range(len(x))]

    def rearrange(self, x, r):
        return [x[ri] for ri in r]


def load_dict():
    with open('data/qa_dict.txt', 'rb') as f:
        qa_dict = pickle.load(f)
    with open('data/qa_reverse_dict.txt', 'rb') as f:
        qa_reverse_dict = pickle.load(f)

    return qa_dict, qa_reverse_dict


def load_train_data():
    with open('data/qtext2id.txt', 'rb') as f:
        qtext2id = pickle.load(f)
    with open('data/atext2id_input.txt', 'rb') as f:
        atext2id_input = pickle.load(f)
    with open('data/atext2id_target.txt', 'rb') as f:
        atext2id_target = pickle.load(f)

    return qtext2id, atext2id_input, atext2id_target


def main(unused_argv):
    qa_dict, qa_reverse_dict = load_dict()

    transformer = Transformer(go=qa_dict['<GO>'], eos=qa_dict['<EOS>'],
                              l_dict_qa=len(qa_dict))

    if FLAGS.mode == 'train0':  # train first time or retrain
        if not os.path.exists(FLAGS.model_save_path):
            os.makedirs(FLAGS.model_save_path)

        transformer.build_model()
        qtext2id, atext2id_input, atext2id_target = load_train_data()
        qtext2id_length = [len(q) for q in qtext2id]
        atext2id_input_length = [len(a) for a in atext2id_input]

        transformer.train(qtext2id, qtext2id_length, atext2id_input, atext2id_target, atext2id_input_length,
                          qa_dict, qa_reverse_dict)
    elif FLAGS.mode == 'train1':  # continue train
        qtext2id, atext2id_input, atext2id_target = load_train_data()
        qtext2id_length = [len(q) for q in qtext2id]
        atext2id_input_length = [len(a) for a in atext2id_input]

        transformer.train(qtext2id, qtext2id_length, atext2id_input, atext2id_target, atext2id_input_length,
                          qa_dict, qa_reverse_dict)
    elif FLAGS.mode == 'predict':
        transformer.predict(qa_dict, qa_reverse_dict)


if __name__ == '__main__':
    tf.app.run()
