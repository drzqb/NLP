'''
    deep cross network for ctr
'''
import numpy as np
import tensorflow as tf
import sys
from build_data import load_data

tf.flags.DEFINE_integer('batch_size', 256, 'batch size for training')
tf.flags.DEFINE_integer('epochs', 1000, 'number of iterations')
tf.flags.DEFINE_integer('embedding_size', 8, 'embedding size for word embedding')
tf.flags.DEFINE_multi_integer('dense_size', [32, 32], 'some Dense Layers''size')
tf.flags.DEFINE_integer('cross_layers', 3, 'number of layers for cross network')
tf.flags.DEFINE_string('model_save_path', 'model/dcn1', 'directory of model file saved')
tf.flags.DEFINE_float('lr', 0.1, 'learning rate for training')
tf.flags.DEFINE_float('keep_prob', 0.5, 'rate for dropout')
tf.flags.DEFINE_float('l2', 0.1, 'rate for dropout')
tf.flags.DEFINE_integer('per_save', 10, 'save model once every per_save iterations')
tf.flags.DEFINE_string('mode', 'train0', 'The mode of train or predict as follows: '
                                         'train0: train first time or retrain'
                                         'train1: continue train'
                                         'predict: predict')

CONFIG = tf.flags.FLAGS

NUMERIC_FIELDS = 35


class DCN():
    def __init__(self, config, feature_sizes, field_size):
        self.config = config
        self.feature_sizes = feature_sizes
        self.field_size = field_size
        self.N = NUMERIC_FIELDS + config.embedding_size * (field_size - NUMERIC_FIELDS)
        self.build_model()

    def build_model(self):
        with tf.name_scope('input'):
            self.feature_index = tf.placeholder(tf.int32, shape=[None, self.field_size], name='feature_index')
            self.feature_value = tf.placeholder(tf.float32, shape=[None, self.field_size], name='feature_value')
            self.label = tf.placeholder(tf.int32, shape=[None], name='label')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.name_scope('embedding'):
            embedding_matrix = tf.Variable(
                tf.random_normal([self.feature_sizes, self.config.embedding_size], 0.0, 0.01))
            feature_embeded = tf.nn.embedding_lookup(embedding_matrix,
                                                     self.feature_index[:, NUMERIC_FIELDS:])
            now = tf.concat([self.feature_value[:, :NUMERIC_FIELDS],
                             tf.reshape(feature_embeded,
                                        [-1, self.config.embedding_size * (self.field_size - NUMERIC_FIELDS)])],
                            axis=-1)

        with tf.name_scope('deepANDcross'):
            now_cross = now
            now_deep = now
            regl2 = tf.contrib.layers.l2_regularizer(self.config.l2)
            for i in range(len(self.config.dense_size)):
                now_deep = tf.nn.dropout(tf.layers.dense(now_deep, self.config.dense_size[i],
                                                         activation='relu',
                                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                         kernel_regularizer=regl2),
                                         keep_prob=self.keep_prob)
            for i in range(self.config.cross_layers):
                now_cross = tf.multiply(now, tf.layers.dense(now_cross, 1, use_bias=False)) + \
                            tf.Variable(tf.random_normal([self.N])) + now_cross

        with tf.name_scope('output'):
            out = tf.squeeze(tf.layers.dense(tf.concat([now_deep, now_cross], axis=-1), 1,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             kernel_regularizer=regl2), axis=-1)
            self.output = tf.sigmoid(out)

        with tf.name_scope('loss'):
            self.prediction = tf.cast(tf.greater_equal(self.output, 0.5), tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.label), tf.float32))
            self.auc = tf.metrics.auc(self.label, self.output)
            # self.loss = tf.reduce_mean(
            #     tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.label, tf.float32), logits=out))
            self.loss = tf.reduce_mean(tf.square(tf.cast(self.label, tf.float32) - self.output))
            self.loss += tf.losses.get_regularization_loss()

        optimizer = tf.train.GradientDescentOptimizer(self.config.lr)
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients = tf.clip_by_global_norm(gradients, 5)[0]
        self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

    def train(self, sess, feature_index, feature_value, label):
        loss, accuracy, auc, _ = sess.run([self.loss, self.accuracy, self.auc, self.train_op], feed_dict={
            self.feature_index: feature_index,
            self.feature_value: feature_value,
            self.label: label,
            self.keep_prob: self.config.keep_prob
        })
        return loss, accuracy, auc

    def eval(self, sess, feature_index, feature_value, label):
        accuracy, auc = sess.run([self.accuracy, self.auc], feed_dict={
            self.feature_index: feature_index,
            self.feature_value: feature_value,
            self.label: label,
            self.keep_prob: 1.0
        })
        return accuracy, auc

    def predict(self, sess, feature_index, feature_value):
        result = sess.run(self.output, feed_dict={
            self.feature_index: feature_index,
            self.feature_value: feature_value,
            self.keep_prob: 1.0
        })
        return result

    def save(self, sess, path):
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver(max_to_keep=1)
        saver.restore(sess, save_path=path)


def get_batch(Xi, Xv, y, batch_size, index):
    start = index * batch_size
    end = (index + 1) * batch_size
    return Xi[start:end], Xv[start:end], np.array(y[start:end])


def main(unused_argvs):
    data = load_data()

    feature_sizes = data['feat_dim']
    field_size = len(data['xi'][0])

    with tf.Session() as sess:
        mydcn = DCN(CONFIG, feature_sizes, field_size)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        m_samples = len(data['y_train'])
        total_batch = m_samples // CONFIG.batch_size
        index = np.array(data['xi'], dtype=np.int32)
        value = np.array(data['xv'], dtype=np.float32)
        label = np.squeeze(np.array(data['y_train'], dtype=np.int32), axis=-1)

        index_val = index[:CONFIG.batch_size]
        value_val = value[:CONFIG.batch_size]
        label_val = label[:CONFIG.batch_size]

        if CONFIG.mode.startswith('train'):
            if CONFIG.mode == 'train1':
                mydcn.restore(sess, CONFIG.model_save_path)

            loss = []
            acc = []
            auc = []
            for epoch in range(1, CONFIG.epochs + 1):
                loss_epoch = 0.0
                acc_epoch = 0.0
                auc_epoch = 0.0

                for batch in range(total_batch):
                    X_index, X_value, y = get_batch(index, value, label, CONFIG.batch_size, batch)
                    loss_batch, acc_batch, auc_batch = mydcn.train(sess, X_index, X_value, y)
                    loss_epoch += loss_batch
                    acc_epoch += acc_batch
                    auc_epoch += auc_batch[0]

                    sys.stdout.write('\r>> %d/%d | %d/%d | loss_batch: %f  acc_batch:%.2f%%  auc_batch:%.2f' % (
                        epoch, CONFIG.epochs, batch + 1, total_batch, loss_batch, 100.0 * acc_batch, auc_batch[0]))
                    sys.stdout.flush()
                loss.append(loss_epoch / total_batch)
                acc.append(acc_epoch / total_batch)
                auc.append(auc_epoch / total_batch)

                sys.stdout.write(' | loss: %f  acc:%.2f%%  auc:%.2f\n' % (loss[-1], 100.0 * acc[-1], auc[-1]))
                sys.stdout.flush()

                acc_val, auc_val = mydcn.eval(sess, index_val, value_val, label_val)
                sys.stdout.write('  acc_val:%.2f%%  auc_val:%.2f\n' % (100.0 * acc_val, auc_val[0]))
                sys.stdout.flush()

                r = np.random.permutation(m_samples)
                index = index[r]
                value = value[r]
                label = label[r]

                if epoch % CONFIG.per_save == 0:
                    mydcn.save(sess, CONFIG.model_save_path)
        else:
            mydcn.restore(sess, CONFIG.model_save_path)

            for batch in range(total_batch):
                X_index, X_value, y = get_batch(index, value, label, CONFIG.batch_size, batch)
                result = mydcn.predict(sess, X_index, X_value)
                print(y)
                print(result)
                print('----------------------------------------------------------------')


if __name__ == '__main__':
    tf.app.run()
