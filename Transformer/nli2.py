'''
    NLI with full Transformer model
'''
import tensorflow as tf
import pickle
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

EMBED_SIZE = 300

tf.flags.DEFINE_integer('maxword', 100, 'max length of any sentences')
tf.flags.DEFINE_integer('block', 2, 'number of Encoder submodel')
tf.flags.DEFINE_integer('head', 10, 'number of multi_head attention')
tf.flags.DEFINE_integer('batch_size', 100, 'batch size for training')
tf.flags.DEFINE_integer('epochs', 100000, 'number of iterations')
tf.flags.DEFINE_float('keep_prob', 0.5, 'keep probility for rnn outputs')

tf.flags.DEFINE_string('model_save_path', 'model/', 'directory of model file saved')
tf.flags.DEFINE_float('lr', 0.0001, 'learning rate for training')
tf.flags.DEFINE_integer('per_save', 1000, 'save model once every per_save iterations')
tf.flags.DEFINE_string('mode', 'train0', 'The mode of train or predict as follows: '
                                         'train0: train first time or retrain'
                                         'train1: continue train'
                                         'predict: predict')
CONFIG = tf.flags.FLAGS


def label_smoothing(inputs, K=3, epsilon=0.1):
    return ((1 - epsilon) * inputs) + (epsilon / K)


def single_example_parser(serialized_example):
    context_features = {
        'label': tf.FixedLenFeature([], tf.int64),
        'p_len': tf.FixedLenFeature([], tf.int64),
        'h_len': tf.FixedLenFeature([], tf.int64)
    }
    sequence_features = {
        'premise': tf.FixedLenSequenceFeature([],
                                              tf.int64),
        'hypothesis': tf.FixedLenSequenceFeature([],
                                                 tf.int64)}

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    l = context_parsed['label']
    p_len = context_parsed['p_len']
    h_len = context_parsed['h_len']

    p = sequence_parsed['premise']
    h = sequence_parsed['hypothesis']
    return p, h, l, p_len, h_len


def batched_data(tfrecord_filename, single_example_parser, batch_size, padded_shapes, buffer_size=2000,
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


class NLI():
    def __init__(self, config, l_dict_len):
        self.config = config
        self.l_dict_len = l_dict_len

    def build_model(self, embed_matrix):
        with tf.name_scope('input'):
            p = tf.placeholder(tf.int32, [None, None], 'premise')
            h = tf.placeholder(tf.int32, [None, None], 'hypothesis')
            label = tf.placeholder(tf.int32, [None], 'label')
            p_len = tf.placeholder(tf.int32, [None], 'p_len')
            h_len = tf.placeholder(tf.int32, [None], 'h_len')
            pos = tf.placeholder(tf.float32, [None, EMBED_SIZE], name='position')
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            max_p_len = tf.reduce_max(p_len, name='max_length_p')
            max_h_len = tf.reduce_max(h_len, name='max_length_h')
            padding_mask_p = tf.tile(tf.expand_dims(tf.greater(p, 0), 1), [self.config.head, tf.shape(p)[1], 1],
                                     name='mask_p')
            padding_mask_h = tf.tile(tf.expand_dims(tf.greater(h, 0), 1), [self.config.head, tf.shape(h)[1], 1],
                                     name='mask_h')
            padding_mask_ph = tf.tile(tf.expand_dims(tf.greater(h, 0), 1), [self.config.head, tf.shape(p)[1], 1], name='mask_ph')
            padding_mask_hp = tf.tile(tf.expand_dims(tf.greater(p, 0), 1), [self.config.head, tf.shape(h)[1], 1], name='mask_hp')

            sequence_mask_p = tf.tile(tf.expand_dims(tf.sequence_mask(p_len, max_p_len), 2), [1, 1, EMBED_SIZE])
            sequence_mask_h = tf.tile(tf.expand_dims(tf.sequence_mask(h_len, max_h_len), 2), [1, 1, EMBED_SIZE])

        with tf.name_scope('embedding'):
            embed_matrix1 = tf.Variable(tf.random_uniform([2, EMBED_SIZE], -1.0, 1.0), dtype=tf.float32)
            embed_matrix2 = tf.Variable(tf.constant(embed_matrix, dtype=tf.float32), trainable=False)
            embed_matrix12 = tf.concat([embed_matrix1, embed_matrix2], axis=0, name='embed_matrix')
            p_embed = tf.nn.embedding_lookup(embed_matrix12, p, name='p_embed')
            h_embed = tf.nn.embedding_lookup(embed_matrix12, h, name='h_embed')

            now_p = tf.nn.dropout(p_embed + pos[:max_p_len], keep_prob)
            now_h = tf.nn.dropout(h_embed + pos[:max_h_len], keep_prob)

        for block in range(self.config.block):
            with tf.variable_scope('selfattention' + str(block)):
                WQ = tf.layers.Dense(EMBED_SIZE, use_bias=False, name='Q')
                WK = tf.layers.Dense(EMBED_SIZE, use_bias=False, name='K')
                WV = tf.layers.Dense(EMBED_SIZE, use_bias=False, name='V')
                WO = tf.layers.Dense(EMBED_SIZE, use_bias=False, name='O')

                Q = tf.concat(tf.split(WQ(now_p), self.config.head, axis=-1), axis=0)
                K = tf.concat(tf.split(WK(now_p), self.config.head, axis=-1), axis=0)
                V = tf.concat(tf.split(WV(now_p), self.config.head, axis=-1), axis=0)

                QK = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(EMBED_SIZE / self.config.head)
                Z_p = tf.nn.dropout(WO(tf.concat(
                    tf.split(tf.matmul(softmax(QK, padding_mask_p), V), self.config.head,
                             axis=0), axis=-1)), keep_prob)

                Q = tf.concat(tf.split(WQ(now_h), self.config.head, axis=-1), axis=0)
                K = tf.concat(tf.split(WK(now_h), self.config.head, axis=-1), axis=0)
                V = tf.concat(tf.split(WV(now_h), self.config.head, axis=-1), axis=0)

                QK = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(EMBED_SIZE / self.config.head)
                Z_h = tf.nn.dropout(WO(tf.concat(
                    tf.split(tf.matmul(softmax(QK, padding_mask_h), V), self.config.head,
                             axis=0), axis=-1)), keep_prob)

                scale_sa = tf.get_variable('scale_sa', initializer=tf.ones([EMBED_SIZE], dtype=tf.float32))
                bias_sa = tf.get_variable('bias_sa', initializer=tf.zeros([EMBED_SIZE], dtype=tf.float32))

                now_p = layer_norm_compute(now_p + Z_p, scale_sa, bias_sa)
                now_h = layer_norm_compute(now_h + Z_h, scale_sa, bias_sa)

            with tf.variable_scope('mutulattention' + str(block)):
                WQ = tf.layers.Dense(EMBED_SIZE, use_bias=False, name='Q')
                WK = tf.layers.Dense(EMBED_SIZE, use_bias=False, name='K')
                WV = tf.layers.Dense(EMBED_SIZE, use_bias=False, name='V')
                WO = tf.layers.Dense(EMBED_SIZE, use_bias=False, name='O')

                Q = tf.concat(tf.split(WQ(now_p), self.config.head, axis=-1), axis=0)
                K = tf.concat(tf.split(WK(now_h), self.config.head, axis=-1), axis=0)
                V = tf.concat(tf.split(WV(now_h), self.config.head, axis=-1), axis=0)

                QK = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(EMBED_SIZE / self.config.head)
                Z_p = tf.nn.dropout(WO(tf.concat(
                    tf.split(tf.matmul(softmax(QK, padding_mask_ph), V), self.config.head,
                             axis=0), axis=-1)), keep_prob)

                Q = tf.concat(tf.split(WQ(now_h), self.config.head, axis=-1), axis=0)
                K = tf.concat(tf.split(WK(now_p), self.config.head, axis=-1), axis=0)
                V = tf.concat(tf.split(WV(now_p), self.config.head, axis=-1), axis=0)

                QK = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(EMBED_SIZE / self.config.head)
                Z_h = tf.nn.dropout(WO(tf.concat(
                    tf.split(tf.matmul(softmax(QK, padding_mask_hp), V), self.config.head,
                             axis=0), axis=-1)), keep_prob)

                scale_ma = tf.get_variable('scale_ma', initializer=tf.ones([EMBED_SIZE], dtype=tf.float32))
                bias_ma = tf.get_variable('bias_ma', initializer=tf.zeros([EMBED_SIZE], dtype=tf.float32))

                now_p = layer_norm_compute(now_p + Z_p, scale_ma, bias_ma)
                now_h = layer_norm_compute(now_h + Z_h, scale_ma, bias_ma)

            with tf.variable_scope('feedforward' + str(block)):
                ffrelu = tf.layers.Dense(4 * EMBED_SIZE, activation=tf.nn.relu, name='ffrelu')
                ff = tf.layers.Dense(EMBED_SIZE, name='ff')
                scale_ff = tf.get_variable('scale_ff', initializer=tf.ones([EMBED_SIZE], dtype=tf.float32))
                bias_ff = tf.get_variable('bias_ff', initializer=tf.zeros([EMBED_SIZE], dtype=tf.float32))

                now_p = layer_norm_compute(ff(ffrelu(now_p)) + now_p, scale_ff, bias_ff)
                now_h = layer_norm_compute(ff(ffrelu(now_h)) + now_h, scale_ff, bias_ff)

        with tf.name_scope('integrate'):
            now_pf = tf.zeros_like(now_p)
            now_p = tf.reduce_sum(tf.where(sequence_mask_p, now_p, now_pf), axis=1)
            now_p = tf.div(now_p, tf.expand_dims(tf.cast(p_len, tf.float32), axis=1))

            now_hf = tf.zeros_like(now_h)
            now_h = tf.reduce_sum(tf.where(sequence_mask_h, now_h, now_hf), axis=1)
            now_h = tf.div(now_h, tf.expand_dims(tf.cast(h_len, tf.float32), axis=1))

        with tf.variable_scope('classify'):
            cph = tf.concat([now_p, now_h], axis=-1)
            logits = tf.identity(tf.layers.dense(cph, self.l_dict_len, name='project'), name='logits')
            prediction = tf.argmax(logits, axis=-1, output_type=tf.int32, name='prediction')

        with tf.name_scope('loss'):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, label), tf.float32), name='accuracy')

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=label_smoothing(tf.one_hot(label, depth=self.l_dict_len)),
                logits=logits),
                name='loss')

            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config.lr, name='optimizer')
            train_op = optimizer.minimize(loss, name='train_op')

        writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph(), filename_suffix='mynli8')
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
        saver.save(sess, self.config.model_save_path + 'mynli8')
        sess.close()
        print('Model saved successfully!')

    def train(self):
        train_file = ['data/train.tfrecord']
        valid_file = ['data/val.tfrecord']

        train_batch = batched_data(train_file, single_example_parser, self.config.batch_size,
                                   padded_shapes=([-1], [-1], [], [], []))
        valid_batch = batched_data(valid_file, single_example_parser, self.config.batch_size,
                                   padded_shapes=([-1], [-1], [], [], []),
                                   shuffle=False)
        sess = tf.Session()

        newsaver = tf.train.import_meta_graph(self.config.model_save_path + 'mynli8.meta')
        newsaver.restore(sess, self.config.model_save_path + 'mynli8')

        graph = tf.get_default_graph()
        p = graph.get_operation_by_name('input/premise').outputs[0]
        h = graph.get_operation_by_name('input/hypothesis').outputs[0]
        label = graph.get_operation_by_name('input/label').outputs[0]
        p_len = graph.get_operation_by_name('input/p_len').outputs[0]
        h_len = graph.get_operation_by_name('input/h_len').outputs[0]
        pos = graph.get_operation_by_name('input/position').outputs[0]

        keep_prob = graph.get_operation_by_name('input/keep_prob').outputs[0]

        accuracy = graph.get_tensor_by_name('loss/accuracy:0')
        loss = graph.get_tensor_by_name('loss/loss:0')
        train_op = graph.get_operation_by_name('loss/train_op')

        pos_enc = np.array(
            [[position / np.power(10000.0, 2.0 * (i // 2) / EMBED_SIZE) for i in range(EMBED_SIZE)]
             for
             position in range(self.config.maxword)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])

        loss_ = []
        acc_ = []
        best_val_acc = 0.0
        for epoch in range(1, self.config.epochs + 1):
            train_batch_ = sess.run(train_batch)

            feed_dict = {p: train_batch_[0],
                         h: train_batch_[1],
                         label: train_batch_[2],
                         p_len: train_batch_[3],
                         h_len: train_batch_[4],
                         pos: pos_enc,
                         keep_prob: self.config.keep_prob
                         }
            loss_batch, _, acc_batch = sess.run([loss, train_op, accuracy], feed_dict=feed_dict)
            loss_.append(loss_batch)
            acc_.append(acc_batch)

            sys.stdout.write('\r>> %d/%d  | loss_batch: %f  acc_batch:%.3f' % (
                epoch, self.config.epochs, loss_batch, acc_batch))
            sys.stdout.flush()

            if epoch % self.config.per_save == 0:
                valid_acc = []
                for i in range(10):
                    valid_batch_ = sess.run(valid_batch)

                    valid_feed_dict = {p: valid_batch_[0],
                                       h: valid_batch_[1],
                                       label: valid_batch_[2],
                                       p_len: valid_batch_[3],
                                       h_len: valid_batch_[4],
                                       pos: pos_enc,
                                       keep_prob: 1.0
                                       }
                    valid_acc.append(sess.run(accuracy, feed_dict=valid_feed_dict))

                avg_valid_acc = np.mean(valid_acc)

                sys.stdout.write(
                    '  train_loss:%f  train_acc:%.5f  | valid_acc:%.5f\n' % (
                        np.mean(loss_[-self.config.per_save:]),
                        np.mean(acc_[-self.config.per_save:]),
                        avg_valid_acc))
                sys.stdout.flush()

                if avg_valid_acc > best_val_acc:
                    newsaver.save(sess, self.config.model_save_path + 'mynli8')
                    print('model saved successfully!')
                    best_val_acc = avg_valid_acc
        sess.close()

        fig = plt.figure(figsize=(10, 8))
        plt.plot(loss_)
        plt.savefig(self.config.model_save_path + 'mynli8_loss.png')
        plt.close(fig)

    def predict(self):
        test_file = ['data/test.tfrecord']

        test_batch = batched_data(test_file, single_example_parser, self.config.batch_size,
                                  padded_shapes=([-1], [-1], [], [], []),
                                  shuffle=False)
        sess = tf.Session()

        newsaver = tf.train.import_meta_graph(self.config.model_save_path + 'mynli8.meta')
        newsaver.restore(sess, self.config.model_save_path + 'mynli8')

        graph = tf.get_default_graph()
        p = graph.get_operation_by_name('input/premise').outputs[0]
        h = graph.get_operation_by_name('input/hypothesis').outputs[0]
        label = graph.get_operation_by_name('input/label').outputs[0]
        p_len = graph.get_operation_by_name('input/p_len').outputs[0]
        h_len = graph.get_operation_by_name('input/h_len').outputs[0]
        pos = graph.get_operation_by_name('input/position').outputs[0]

        keep_prob = graph.get_operation_by_name('input/keep_prob').outputs[0]

        accuracy = graph.get_tensor_by_name('loss/accuracy:0')

        pos_enc = np.array(
            [[position / np.power(10000.0, 2.0 * (i // 2) / EMBED_SIZE) for i in range(EMBED_SIZE)]
             for
             position in range(self.config.maxword)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])

        test_acc = []
        for i in range(10):
            test_batch_ = sess.run(test_batch)

            test_feed_dict = {p: test_batch_[0],
                              h: test_batch_[1],
                              label: test_batch_[2],
                              p_len: test_batch_[3],
                              h_len: test_batch_[4],
                              pos: pos_enc,
                              keep_prob: 1.0
                              }
            test_acc.append(sess.run(accuracy, feed_dict=test_feed_dict))

        print('\033[1;31;40m')
        sys.stdout.write('      test_acc:%.3f\n' % (np.mean(test_acc)))
        sys.stdout.flush()
        print('\033[0m')

        sess.close()


def main(unused_argv):
    with open('data/label_dict.txt', 'rb') as f:
        l_dict_len = len(pickle.load(f))

    with open('data/embed_matrix.txt', 'rb') as f:
        embed_matrix = pickle.load(f)

    nli = NLI(CONFIG, l_dict_len)
    if CONFIG.mode == 'train0':
        if not os.path.exists(CONFIG.model_save_path):
            os.makedirs(CONFIG.model_save_path)
        nli.build_model(embed_matrix)
        nli.train()
    elif CONFIG.mode == 'train1':
        nli.train()
    elif CONFIG.mode == 'predict':
        nli.predict()


if __name__ == '__main__':
    tf.app.run()
