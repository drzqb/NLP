'''
    GRU for speech recognition
'''
import tensorflow as tf
import numpy as np
import sys
import os
import librosa
import matplotlib.pyplot as plt
import time
from sr_hp import CONFIG


class Path:
    wav_train = 'train/'
    wav_test = 'test/'
    wav_validation = 'validation/'
    mfcc = 'mfcc/'
    model = 'model/'


class Txt:
    title = 'Speech Recognition'


def dense_to_one_hot(label):
    return np.eye(CONFIG.n_classes)[label]


def load_wav(path):
    features = []
    length = []
    labels = []
    files = os.listdir(path)

    for wav in files:
        if not wav.endswith(".mp3"): continue
        wave, sr = librosa.load(path + wav, mono=True)
        label = dense_to_one_hot(int(wav[0]))
        labels.append(label)
        mfcc = librosa.feature.mfcc(wave, sr,n_mfcc=40)
        length.append(len(mfcc[0]))
        mfcc = np.pad(mfcc, ((0, 0), (0, 60 - len(mfcc[0]))), mode='constant', constant_values=0)
        features.append(np.array(mfcc))
    return np.array(features).transpose([0, 2, 1]), np.array(labels, dtype=np.int), np.array(length, dtype=np.int)


def load_speech():
    if not os.path.exists(Path.mfcc + 'trainX.npy'):
        print('loding wav files...')
        trainX, trainY, trainLen = load_wav(Path.wav_train)
        valX, valY, valLen = load_wav(Path.wav_validation)
        Xmean = np.mean(trainX, axis=(0, 1))
        Xvar = np.var(trainX, axis=(0, 1))

        np.save(Path.mfcc + 'trainX.npy', trainX)
        np.save(Path.mfcc + 'trainY.npy', trainY)
        np.save(Path.mfcc + 'trainLen.npy', trainLen)
        np.save(Path.mfcc + 'valX.npy', valX)
        np.save(Path.mfcc + 'valY.npy', valY)
        np.save(Path.mfcc + 'valLen.npy', valLen)
        np.save(Path.mfcc + 'Xmean.npy', Xmean)
        np.save(Path.mfcc + 'Xvar.npy', Xvar)
    else:
        print('loading train and val data...')
        trainX = np.load(Path.mfcc + 'trainX.npy')
        trainY = np.load(Path.mfcc + 'trainY.npy')
        trainLen = np.load(Path.mfcc + 'trainLen.npy')
        valX = np.load(Path.mfcc + 'valX.npy')
        valY = np.load(Path.mfcc + 'valY.npy')
        valLen = np.load(Path.mfcc + 'valLen.npy')
        Xmean = np.load(Path.mfcc + 'Xmean.npy')
        Xvar = np.load(Path.mfcc + 'Xvar.npy')

    trainX = ((trainX - Xmean) / np.sqrt(Xvar + 1.0e-10))
    valX = ((valX - Xmean) / np.sqrt(Xvar + 1.0e-10))

    return trainX, trainY, trainLen, valX, valY, valLen


def build_model():
    def GRU(x, l, p_keep):
        def gcell(p_keep):
            cell = tf.nn.rnn_cell.GRUCell(CONFIG.n_hidden_units)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=p_keep)
            return cell

        mcell = tf.nn.rnn_cell.MultiRNNCell([gcell(p_keep) for _ in range(CONFIG.n_layers)],
                                            state_is_tuple=True)  # 多层lstm cell 堆叠起来

        _, final_state = tf.nn.dynamic_rnn(mcell, inputs=x, sequence_length=l, dtype=tf.float32)

        return final_state[-1]

    # tf Graph input
    with tf.name_scope('Input'):
        x = tf.placeholder(tf.float32, [None, None, 40], name='X')
        y = tf.placeholder(tf.float32, [None, CONFIG.n_classes], name='Y')
        l = tf.placeholder(tf.int32, [None], name='L')
        lr = tf.placeholder(tf.float32, name='lr')
        p_keep = tf.placeholder(tf.float32, name='p_keep')

    # Define weights
    with tf.name_scope('GRU'):
        pred = tf.layers.dense(GRU(x, l, p_keep), CONFIG.n_classes)

    with tf.name_scope('Train'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y), name='loss')
        train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss, name='train_op')

        pred_op = tf.argmax(pred, 1, name='predict')
        correct_pred = tf.equal(pred_op, tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    writer = tf.summary.FileWriter('logs/', graph=tf.get_default_graph(), filename_suffix='gru')
    writer.flush()
    writer.close()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=1)
    if not os.path.exists(Path.model):
        os.makedirs(Path.model)

    saver.save(sess, Path.model + 'gru')
    print('model saved successfully')


def train():
    sess = tf.Session()
    newsaver = tf.train.import_meta_graph(Path.model + 'gru.meta')
    newsaver.restore(sess, Path.model + 'gru')

    graph = tf.get_default_graph()

    x = graph.get_operation_by_name('Input/X').outputs[0]
    y = graph.get_operation_by_name('Input/Y').outputs[0]
    l = graph.get_operation_by_name('Input/L').outputs[0]
    lr = graph.get_operation_by_name('Input/lr').outputs[0]
    p_keep = graph.get_operation_by_name('Input/p_keep').outputs[0]

    loss = graph.get_tensor_by_name('Train/loss:0')
    train_op = graph.get_operation_by_name('Train/train_op')
    accuracy = graph.get_tensor_by_name('Train/accuracy:0')

    X_train, y_train, X_train_len, X_val, y_val, X_val_len = load_speech()

    m_samples = np.shape(X_train)[0]

    print('Training...')
    loss_all = []
    accuracy_all = []

    decay_steps = m_samples // CONFIG.batch_size

    for i in range(CONFIG.epochs):
        mini = np.array_split(np.random.permutation(m_samples), decay_steps)

        temp_loss = 0.0
        for idx in mini:
            tmp, _ = sess.run([loss, train_op],
                              feed_dict={x: X_train[idx], y: y_train[idx], l: X_train_len[idx], p_keep: CONFIG.p_keep,
                                         lr: CONFIG.lr})
            temp_loss += len(idx) * tmp
        loss_all.append(temp_loss / m_samples)
        accuracy_all.append(sess.run(accuracy, feed_dict={x: X_val, y: y_val, l: X_val_len, p_keep: 1.0}))
        print(i + 1, '|   loss:%.9f' % loss_all[-1], '    accuracy:%.1f%%' % (100.0 * accuracy_all[-1]),
              '  %s' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        if (i + 1) % CONFIG.per_save == 0:
            newsaver.save(sess, Path.model + 'SR')
            print('model saved successfully')

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(loss_all, 'r', label='loss')
    ax1.set_ylabel('Loss')
    ax1.legend(loc=(.7, .5), fontsize=10, shadow=True)
    ax2 = ax1.twinx()
    ax2.plot(accuracy_all, 'g', label='accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc=(.7, .6), fontsize=10, shadow=True)
    plt.title(Txt.title)
    plt.show()


def predict():
    X_test, y_test, X_test_len = load_wav(Path.wav_test)

    Xmean = np.load(Path.mfcc + 'Xmean.npy')
    Xvar = np.load(Path.mfcc + 'Xvar.npy')

    X_test = ((X_test - Xmean) / np.sqrt(Xvar + 1.0e-10))

    with tf.Session() as sess:
        tf.train.import_meta_graph(Path.model + 'SR.meta').restore(sess, Path.model + 'SR')
        x = tf.get_default_graph().get_operation_by_name('Input/X').outputs[0]
        y = tf.get_default_graph().get_operation_by_name('Input/Y').outputs[0]
        l = tf.get_default_graph().get_operation_by_name('Input/L').outputs[0]
        p_keep = tf.get_default_graph().get_operation_by_name('Input/p_keep').outputs[0]

        predict = tf.get_default_graph().get_tensor_by_name('Train/predict:0')
        accuracy = tf.get_default_graph().get_tensor_by_name('Train/accuracy:0')
        predict_, accuracy_ = sess.run([predict, accuracy],
                                       feed_dict={x: X_test, y: y_test, l: X_test_len, p_keep: 1.0})
        real = np.argmax(y_test, axis=1)

        for i in range(np.shape(X_test)[0]):
            print(predict_[i], '--->', real[i])

        print('Accuracy for test: %.2f%%' % (100.0 * accuracy_))


if __name__ == '__main__':
    if CONFIG.mode == 'train0':
        build_model()
        train()
    elif CONFIG.mode == 'train1':
        train()
    elif CONFIG.mode == 'predict':
        predict()
