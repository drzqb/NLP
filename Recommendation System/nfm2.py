'''
    Neural Factorization Machine (NFM) for CTR
    Serial structure
    ResNet structure for Deep
'''
import numpy as np
import tensorflow as tf
import sys
from build_data import load_data

tf.flags.DEFINE_integer('batch_size', 100, 'batch size for training')
tf.flags.DEFINE_integer('epochs', 1000, 'number of iterations')
tf.flags.DEFINE_integer('embedding_size', 256, 'embedding size for word embedding')
tf.flags.DEFINE_integer('dense_layer', 10, 'number of resnet dense layers')
tf.flags.DEFINE_string('model_save_path', 'model/nfm2', 'directory of model file saved')
tf.flags.DEFINE_float('lr', 0.1, 'learning rate for training')
tf.flags.DEFINE_integer('per_save', 10, 'save model once every per_save iterations')
tf.flags.DEFINE_string('mode', 'train0', 'The mode of train or predict as follows: '
                                         'train0: train first time or retrain'
                                         'train1: continue train'
                                         'predict: predict')

CONFIG = tf.flags.FLAGS


class NFM():
    def __init__(self, config, feature_sizes, field_size):
        self.config = config
        self.feature_sizes = feature_sizes
        self.field_size = field_size
        self.build_model()

    def build_model(self):
        with tf.name_scope('input'):
            self.feature_index = tf.placeholder(tf.int32, shape=[None, None], name='feature_index')
            self.feature_value = tf.placeholder(tf.float32, shape=[None, None], name='feature_value')
            self.label = tf.placeholder(tf.int32, shape=[None], name='label')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.name_scope('embedding'):
            embedding_matrix1 = tf.Variable(tf.random_normal([self.feature_sizes, 1], 0.0, 1.0))
            feature_embeded1 = tf.nn.embedding_lookup(embedding_matrix1, self.feature_index)
            embedding_matrix2 = tf.Variable(
                tf.random_normal([self.feature_sizes, self.config.embedding_size], 0.0, 0.01))
            feature_embeded2 = tf.nn.embedding_lookup(embedding_matrix2, self.feature_index)

        with tf.name_scope('firstorder'):
            output1 = tf.reduce_sum(tf.squeeze(feature_embeded1, axis=-1), axis=-1, keepdims=True)

        with tf.name_scope('secondorder'):
            feature_mul_value = tf.multiply(feature_embeded2, tf.expand_dims(self.feature_value, axis=2))
            now = 0.5 * (
                    tf.square(tf.reduce_sum(feature_mul_value, axis=1)) - tf.reduce_sum(tf.square(feature_mul_value),
                                                                                        axis=1))
        # ResNet block
        with tf.name_scope('deep'):
            for i in range(self.config.dense_layer):
                now1 = tf.layers.dense(now, self.config.embedding_size,
                                       activation=tf.nn.relu,
                                       # kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                       kernel_initializer=tf.orthogonal_initializer()
                                       )
                now = now + tf.layers.dense(now1, self.config.embedding_size,
                                            activation=tf.nn.relu,
                                            # kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                            kernel_initializer=tf.orthogonal_initializer()
                                            )
            outputdeep = now

        with tf.name_scope('output'):
            out = tf.squeeze(output1 + tf.layers.dense(outputdeep, 1,
                                                       # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                       kernel_initializer=tf.orthogonal_initializer()),
                             axis=-1)
            self.output = tf.sigmoid(out)

        with tf.name_scope('loss'):
            self.prediction = tf.cast(tf.greater_equal(self.output, 0.5), tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.label), tf.float32))
            self.auc = tf.metrics.auc(self.label, self.output)
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.label, tf.float32), logits=out))

        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(self.config.lr, global_step, 1000, 0.99, staircase=True)

        optimizer = tf.train.GradientDescentOptimizer(lr)
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients = tf.clip_by_global_norm(gradients, 5)[0]
        self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params), global_step=global_step)

    def train(self, sess, feature_index, feature_value, label):
        loss, accuracy, auc, _ = sess.run([self.loss, self.accuracy, self.auc, self.train_op], feed_dict={
            self.feature_index: feature_index,
            self.feature_value: feature_value,
            self.label: label,
        })
        return loss, accuracy, auc

    def eval(self, sess, feature_index, feature_value, label):
        accuracy, auc = sess.run([self.accuracy, self.auc], feed_dict={
            self.feature_index: feature_index,
            self.feature_value: feature_value,
            self.label: label,
        })
        return accuracy, auc

    def predict(self, sess, feature_index, feature_value):
        result = sess.run(self.output, feed_dict={
            self.feature_index: feature_index,
            self.feature_value: feature_value,
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
        mynfm = NFM(CONFIG, feature_sizes, field_size)

        # init variables
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
                mynfm.restore(sess, CONFIG.model_save_path)

            loss = []
            acc = []
            auc = []
            for epoch in range(1, CONFIG.epochs + 1):
                loss_epoch = 0.0
                acc_epoch = 0.0
                auc_epoch = 0.0

                for batch in range(total_batch):
                    X_index, X_value, y = get_batch(index, value, label, CONFIG.batch_size, batch)

                    loss_batch, acc_batch, auc_batch = mynfm.train(sess, X_index, X_value, y)
                    loss_epoch += loss_batch
                    acc_epoch += acc_batch
                    auc_epoch += auc_batch[0]

                    sys.stdout.write('\r>> %d/%d | %d/%d | loss_batch: %f  acc_batch:%.2f%%  auc_batch:%.2f' % (
                        epoch, CONFIG.epochs, batch + 1, total_batch, loss_batch, 100.0 * acc_batch, auc_batch[0]))
                    sys.stdout.flush()
                loss.append(loss_epoch / total_batch)
                acc.append(acc_epoch / total_batch)
                auc.append(auc_epoch / total_batch)

                sys.stdout.write(' | loss: %f  acc: %.2f%%  auc: %.2f\n' % (loss[-1], 100.0 * acc[-1], auc[-1]))
                sys.stdout.flush()

                acc_val, auc_val = mynfm.eval(sess, index_val, value_val, label_val)
                sys.stdout.write('  acc_val: %.2f%%  auc_val: %.2f\n' % (100.0 * acc_val, auc_val[0]))
                sys.stdout.flush()

                r = np.random.permutation(m_samples)
                index = index[r]
                value = value[r]
                label = label[r]

                if epoch % CONFIG.per_save == 0:
                    mynfm.save(sess, CONFIG.model_save_path)
        else:
            mynfm.restore(sess, CONFIG.model_save_path)

            for batch in range(total_batch):
                X_index, X_value, y = get_batch(index, value, label, CONFIG.batch_size, batch)
                result = mynfm.predict(sess, X_index, X_value)
                print(y)
                print(result)
                print('----------------------------------------------------------------')


if __name__ == '__main__':
    tf.app.run()
