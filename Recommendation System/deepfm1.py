import numpy as np
import tensorflow as tf
import sys
from build_data import load_data

tf.flags.DEFINE_integer('batch_size', 100, 'batch size for training')
tf.flags.DEFINE_integer('epochs', 100, 'number of iterations')
tf.flags.DEFINE_integer('embedding_size', 256, 'embedding size for word embedding')
tf.flags.DEFINE_multi_integer('dense_size', [512, 256, 128], 'some Dense Layers''size')
tf.flags.DEFINE_string('model_save_path', 'model/deepfm1', 'directory of model file saved')
tf.flags.DEFINE_float('lr', 0.01, 'learning rate for training')
tf.flags.DEFINE_float('keep_prob', 0.5, 'rate for dropout')
tf.flags.DEFINE_integer('per_save', 10, 'save model once every per_save iterations')
tf.flags.DEFINE_string('mode', 'train0', 'The mode of train or predict as follows: '
                                         'train0: train first time or retrain'
                                         'train1: continue train'
                                         'predict: predict')

CONFIG = tf.flags.FLAGS


class DeepFM():
    def __init__(self, config, feature_sizes, field_size):
        self.config = config
        self.feature_sizes = feature_sizes
        self.field_size = field_size
        self.build_model()

    def build_model(self):
        with tf.name_scope('input'):
            self.feature_index = tf.placeholder(tf.int32, shape=[None, None], name='feature_index')
            self.feature_value = tf.placeholder(tf.float32, shape=[None, None], name='feature_value')
            self.label = tf.placeholder(tf.float32, shape=[None], name='label')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.name_scope('embedding'):
            embedding_matrix = tf.Variable(
                tf.random_uniform([self.feature_sizes, self.config.embedding_size], -1.0, 1.0))
            feature_embeded = tf.nn.embedding_lookup(embedding_matrix, self.feature_index)

        with tf.name_scope('firstorder'):
            w = tf.Variable(tf.random_uniform([self.field_size, 1], -1.0, 1.0))
            output1 = tf.squeeze(tf.matmul(self.feature_value, w), axis=-1)

        with tf.name_scope('secondorder'):
            feature_mul_value = tf.multiply(feature_embeded, tf.tile(tf.expand_dims(self.feature_value, axis=2),
                                                                     [1, 1, self.config.embedding_size]))
            output2 = 0.5 * tf.reduce_sum(
                tf.square(tf.reduce_sum(feature_mul_value, axis=1)) - tf.reduce_sum(tf.square(feature_mul_value),
                                                                                    axis=1), axis=1)

        with tf.name_scope('deep'):
            now = tf.reshape(feature_embeded, [-1, self.config.embedding_size * self.field_size])

            for i in range(len(self.config.dense_size)):
                now = tf.nn.dropout(tf.layers.dense(now, self.config.dense_size[i], activation=tf.nn.relu),
                                    keep_prob=self.keep_prob)

            outputdeep = tf.squeeze(tf.layers.dense(now, 1), axis=-1)

        with tf.name_scope('output'):
            output = output1 + output2 + outputdeep
            self.out = tf.nn.sigmoid(output)

        with tf.name_scope('loss'):
            result = tf.cast(tf.greater_equal(self.out, 0.5), tf.float32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(result, self.label), tf.float32), name='accuracy')

            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=output),
                                       name='loss')

        optimizer = tf.train.GradientDescentOptimizer(self.config.lr)
        self.train_op = optimizer.minimize(self.loss, name='train_op')

    def train(self, sess, feature_index, feature_value, label):
        loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.train_op], feed_dict={
            self.feature_index: feature_index,
            self.feature_value: feature_value,
            self.label: label,
            self.keep_prob: self.config.keep_prob
        })
        return loss, accuracy

    def predict(self, sess, feature_index, feature_value):
        result = sess.run(self.out, feed_dict={
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
        mydf = DeepFM(CONFIG, feature_sizes, field_size)

        # init variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if CONFIG.mode.startswith('train'):
            m_samples = len(data['y_train'])
            total_batch = m_samples // CONFIG.batch_size
            index = data['xi']
            value = data['xv']
            label = data['y_train']

            if CONFIG.mode == 'train1':
                mydf.restore(sess, CONFIG.model_save_path)

            loss = []
            acc = []
            for epoch in range(1, CONFIG.epochs + 1):
                loss_epoch = 0.0
                acc_epoch = 0.0

                for batch in range(total_batch):
                    X_index, X_value, y = get_batch(index, value, label, CONFIG.batch_size, batch)

                    loss_batch, acc_batch = mydf.train(sess, X_index, X_value, np.squeeze(y, axis=-1))
                    loss_epoch += loss_batch
                    acc_epoch += acc_batch

                    sys.stdout.write('\r>> %d/%d | %d/%d | loss_batch: %f  acc_batch:%.2f%%' % (
                        epoch, CONFIG.epochs, batch + 1, total_batch, loss_batch, 100.0 * acc_batch))
                    sys.stdout.flush()
                loss.append(loss_epoch / total_batch)
                acc.append(acc_epoch / total_batch)

                sys.stdout.write(' | loss: %f  acc:%.2f%%\n' % (loss[-1], 100.0 * acc[-1]))
                sys.stdout.flush()

                r = np.random.permutation(m_samples)
                index = index[r]
                value = value[r]
                label = label[r]

                if epoch % CONFIG.per_save == 0:
                    mydf.save(sess, CONFIG.model_save_path)
        else:
            mydf.restore(sess, CONFIG.model_save_path)

            m_samples = len(data['y_train'])
            total_batch = m_samples // CONFIG.batch_size
            for batch in range(total_batch):
                X_index, X_value, y = get_batch(data['xi'], data['xv'], data['y_train'], CONFIG.batch_size, batch)
                result = mydf.predict(sess, X_index, X_value)
                print(y)
                print(result)
                print('----------------------------------------------------------------')


if __name__ == '__main__':
    tf.app.run()
