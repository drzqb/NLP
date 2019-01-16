'''
    Decomposable Attention model for NLI
    embeded with selfattention similar to Transformer just to find whether to improve the result...
'''
import tensorflow as tf
import pickle
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

EMBED_SIZE = 300

tf.flags.DEFINE_integer('batch_size', 100, 'batch size for training')
tf.flags.DEFINE_integer('epochs', 100000, 'number of iterations')
tf.flags.DEFINE_integer('rnn_size', 64, 'size of feedforward net F')
tf.flags.DEFINE_integer('rnn_layer', 3, 'number of layers for rnn')
tf.flags.DEFINE_float('keep_prob', 0.5, 'keep probility for rnn outputs')
tf.flags.DEFINE_integer('G_size', 128, 'size of feedforward net G')
tf.flags.DEFINE_integer('G_layer', 3, 'number of Dense G layers')
tf.flags.DEFINE_integer('H_size', 128, 'size of feedforward net H')
tf.flags.DEFINE_integer('H_layer', 3, 'number of Dense H layers')
tf.flags.DEFINE_string('model_save_path', 'model/', 'directory of model file saved')
tf.flags.DEFINE_float('lr', 0.001, 'learning rate for training')
tf.flags.DEFINE_float('grad_clip', 1.0, 'clip for grad based on norm')
tf.flags.DEFINE_integer('per_save', 1000, 'save model once every per_save iterations')
tf.flags.DEFINE_string('mode', 'train0', 'The mode of train or predict as follows: '
                                         'train0: train first time or retrain'
                                         'train1: continue train'
                                         'predict: predict')
CONFIG = tf.flags.FLAGS


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


def batched_data(tfrecord_filename, single_example_parser, batch_size, padded_shapes, buffer_size=10000,
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


class NLI():
    def __init__(self, config, l_dict_len):
        self.config = config
        self.l_dict_len = l_dict_len

    def build_model(self, embed_matrix):
        with tf.name_scope('input'):
            p = tf.placeholder(tf.int32, [None, None], 'premise')
            h = tf.placeholder(tf.int32, [None, None], 'hypothesis')
            l = tf.placeholder(tf.int32, [None], 'label')
            p_len = tf.placeholder(tf.int32, [None], 'p_len')
            h_len = tf.placeholder(tf.int32, [None], 'h_len')
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.name_scope('embedding'):
            embed_matrix1 = tf.Variable(tf.random_uniform([2, EMBED_SIZE], -1.0, 1.0),
                                        dtype=tf.float32)
            embed_matrix2 = tf.Variable(tf.constant(embed_matrix, dtype=tf.float32), trainable=False)
            embed_matrix12 = tf.concat([embed_matrix1, embed_matrix2], axis=0, name='embed_matrix')
            p_embed = tf.nn.embedding_lookup(embed_matrix12, p, name='p_embed')
            h_embed = tf.nn.embedding_lookup(embed_matrix12, h, name='h_embed')

        with tf.name_scope('selfattention'):
            selfattmatrix = tf.Variable(tf.random_uniform([EMBED_SIZE, EMBED_SIZE], -1.0, 1.0, dtype=tf.float32))
            scale = tf.Variable(tf.ones([EMBED_SIZE], dtype=tf.float32))
            bias = tf.Variable(tf.zeros([EMBED_SIZE], dtype=tf.float32))

            j = tf.constant(0, tf.int32)
            p_self = tf.constant(0.0, shape=[0, EMBED_SIZE])
            h_self = tf.constant(0.0, shape=[0, EMBED_SIZE])

            def cond_self(j, p_self, h_self):
                return tf.less(j, tf.shape(p)[0])

            def body_self(j, p_self, h_self):
                attention_p = tf.matmul(tf.matmul(p_embed[j, :p_len[j]], selfattmatrix / EMBED_SIZE),
                                        tf.transpose(p_embed[j, :p_len[j]], [1, 0]))
                attention_h = tf.matmul(tf.matmul(h_embed[j, :h_len[j]], selfattmatrix / EMBED_SIZE),
                                        tf.transpose(h_embed[j, :h_len[j]], [1, 0]))

                p_self = tf.concat([p_self, tf.matmul(tf.nn.softmax(attention_p, axis=-1), p_embed[j, :p_len[j]])],
                                   axis=0)
                h_self = tf.concat([h_self, tf.matmul(tf.nn.softmax(attention_h, axis=-1), h_embed[j, :h_len[j]])],
                                   axis=0)

                kp = tf.constant(0, tf.int32)

                def cond_pad_p(kp, p_self):
                    return tf.less(kp, tf.shape(p)[1] - p_len[j])

                def body_pad_p(kp, p_self):
                    p_self = tf.concat([p_self, tf.constant(0.0, shape=[1, EMBED_SIZE])], axis=0)
                    return tf.add(kp, 1), p_self

                _, p_self = tf.while_loop(cond_pad_p, body_pad_p, [kp, p_self],
                                          shape_invariants=[kp.get_shape(),
                                                            tf.TensorShape([None, EMBED_SIZE])])

                kh = tf.constant(0, tf.int32)

                def cond_pad_h(kh, h_self):
                    return tf.less(kh, tf.shape(h)[1] - h_len[j])

                def body_pad_h(kh, h_self):
                    h_self = tf.concat([h_self, tf.constant(0.0, shape=[1, EMBED_SIZE])], axis=0)
                    return tf.add(kh, 1), h_self

                _, h_self = tf.while_loop(cond_pad_h, body_pad_h, [kh, h_self],
                                          shape_invariants=[kh.get_shape(),
                                                            tf.TensorShape([None, EMBED_SIZE])])

                return tf.add(j, 1), p_self, h_self

            _, p_self, h_self = tf.while_loop(cond_self, body_self, [j, p_self, h_self],
                                              shape_invariants=[j.get_shape(),
                                                                tf.TensorShape([None, EMBED_SIZE]),
                                                                tf.TensorShape([None, EMBED_SIZE])])
            p_self = tf.reshape(p_self, [-1, tf.shape(p)[1], EMBED_SIZE])
            h_self = tf.reshape(h_self, [-1, tf.shape(h)[1], EMBED_SIZE])

            p_se = layer_norm_compute(p_embed + p_self, scale, bias)

            h_se = layer_norm_compute(h_embed + h_self, scale, bias)

        with tf.name_scope('birnn'):
            rnn_fw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(self.config.rnn_size, name='fw' + str(i)), output_keep_prob=keep_prob) for i in
                range(self.config.rnn_layer)])
            rnn_bw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(self.config.rnn_size, name='bw' + str(i)), output_keep_prob=keep_prob) for i in
                range(self.config.rnn_layer)])

            p_outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_fw,
                                                           rnn_bw,
                                                           p_se,
                                                           sequence_length=p_len,
                                                           dtype=tf.float32)
            p_bar = tf.concat(p_outputs, 2)
            h_outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_fw,
                                                           rnn_bw,
                                                           h_se,
                                                           sequence_length=h_len,
                                                           dtype=tf.float32)
            h_bar = tf.concat(h_outputs, 2)

        with tf.name_scope('attention'):
            attmatrix = tf.matmul(p_bar, tf.transpose(h_bar, [0, 2, 1]), name='attmatrix')
            j = tf.constant(0, tf.int32)
            pv = tf.constant(0.0, shape=[0, self.config.G_size])
            hv = tf.constant(0.0, shape=[0, self.config.G_size])

            G = [tf.layers.Dense(self.config.G_size, activation=tf.nn.relu, name='G%d' % i) for i in
                 range(self.config.G_layer)]

            def cond(j, pv, hv):
                return tf.less(j, tf.shape(p)[0])

            def body(j, pv, hv):
                e = attmatrix[j, :p_len[j], :h_len[j]]

                h_weight = tf.nn.softmax(e, axis=-1)
                p_weight = tf.nn.softmax(tf.transpose(e), axis=-1)

                p_wave = tf.matmul(h_weight, h_embed[j, :h_len[j]])
                h_wave = tf.matmul(p_weight, p_embed[j, :p_len[j]])

                cpv = tf.concat([p_embed[j, :p_len[j]], p_wave], axis=-1)
                chv = tf.concat([h_embed[j, :h_len[j]], h_wave], axis=-1)
                for i in range(self.config.G_layer):
                    cpv = G[i](cpv)
                    chv = G[i](chv)

                jpv = tf.expand_dims(tf.reduce_mean(cpv, axis=0), axis=0)
                jhv = tf.expand_dims(tf.reduce_mean(chv, axis=0), axis=0)

                pv = tf.concat([pv, jpv], axis=0)
                hv = tf.concat([hv, jhv], axis=0)
                return tf.add(j, 1), pv, hv

            _, pv, hv = tf.while_loop(cond, body, [j, pv, hv], [j.get_shape(),
                                                                tf.TensorShape([None, self.config.G_size]),
                                                                tf.TensorShape([None, self.config.G_size])])
        with tf.name_scope('classify'):
            H = [tf.layers.Dense(self.config.H_size, activation=tf.nn.relu, name='H%d' % i) for i in
                 range(self.config.H_layer)]

            cph = tf.concat([pv, hv, tf.abs(pv - hv)], axis=-1)
            for i in range(self.config.H_layer):
                cph = H[i](cph)

            logits = tf.identity(tf.layers.dense(cph, self.l_dict_len, name='project'), name='logits')
            prediction = tf.argmax(logits, axis=-1, output_type=tf.int32, name='prediction')

        with tf.name_scope('loss'):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, l), tf.float32), name='accuracy')

            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=l, logits=logits),
                                  name='loss')

            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config.lr, name='optimizer')
            train_variables = tf.trainable_variables()
            grads_vars = optimizer.compute_gradients(loss, train_variables)
            clipgrad = [(tf.clip_by_norm(grad, self.config.grad_clip), var) for grad, var in grads_vars if
                        grad is not None]
            train_op = optimizer.apply_gradients(clipgrad, name='train_op')

        writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph(), filename_suffix='mynli')
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
        saver.save(sess, self.config.model_save_path + 'mynli')
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

        newsaver = tf.train.import_meta_graph(self.config.model_save_path + 'mynli.meta')
        newsaver.restore(sess, self.config.model_save_path + 'mynli')

        graph = tf.get_default_graph()
        p = graph.get_operation_by_name('input/premise').outputs[0]
        h = graph.get_operation_by_name('input/hypothesis').outputs[0]
        l = graph.get_operation_by_name('input/label').outputs[0]
        p_len = graph.get_operation_by_name('input/p_len').outputs[0]
        h_len = graph.get_operation_by_name('input/h_len').outputs[0]
        keep_prob = graph.get_operation_by_name('input/keep_prob').outputs[0]

        accuracy = graph.get_tensor_by_name('loss/accuracy:0')
        loss = graph.get_tensor_by_name('loss/loss:0')
        train_op = graph.get_operation_by_name('loss/train_op')

        loss_ = []
        acc_ = []
        best_val_acc = 0.0
        for epoch in range(1, self.config.epochs + 1):
            train_batch_ = sess.run(train_batch)

            feed_dict = {p: train_batch_[0],
                         h: train_batch_[1],
                         l: train_batch_[2],
                         p_len: train_batch_[3],
                         h_len: train_batch_[4],
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
                                       l: valid_batch_[2],
                                       p_len: valid_batch_[3],
                                       h_len: valid_batch_[4],
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
                    newsaver.save(sess, self.config.model_save_path + 'mynli')
                    print('model saved successfully!')
                    best_val_acc = avg_valid_acc
        sess.close()

        fig = plt.figure(figsize=(10, 8))
        plt.plot(loss_)
        plt.savefig(self.config.model_save_path + 'mynli_loss.png')
        plt.close(fig)

    def predict(self):
        test_file = ['data/test.tfrecord']

        test_batch = batched_data(test_file, single_example_parser, self.config.batch_size,
                                  padded_shapes=([-1], [-1], [], [], []),
                                  shuffle=False)
        sess = tf.Session()

        newsaver = tf.train.import_meta_graph(self.config.model_save_path + 'mynli.meta')
        newsaver.restore(sess, self.config.model_save_path + 'mynli')

        graph = tf.get_default_graph()
        p = graph.get_operation_by_name('input/premise').outputs[0]
        h = graph.get_operation_by_name('input/hypothesis').outputs[0]
        l = graph.get_operation_by_name('input/label').outputs[0]
        p_len = graph.get_operation_by_name('input/p_len').outputs[0]
        h_len = graph.get_operation_by_name('input/h_len').outputs[0]
        keep_prob = graph.get_operation_by_name('input/keep_prob').outputs[0]

        accuracy = graph.get_tensor_by_name('loss/accuracy:0')

        test_acc = []
        for i in range(10):
            test_batch_ = sess.run(test_batch)

            test_feed_dict = {p: test_batch_[0],
                              h: test_batch_[1],
                              l: test_batch_[2],
                              p_len: test_batch_[3],
                              h_len: test_batch_[4],
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
