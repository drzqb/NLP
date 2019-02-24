'''
    seq2seq with attention
             and inference
             and two gru layers
             for chatbot
'''
import numpy as np
import tensorflow as tf
import sys, os
import jieba
import matplotlib.pyplot as plt
import pickle

tf.flags.DEFINE_string('model_save_path', 'model/', 'The path where model shall be saved')
tf.flags.DEFINE_integer('batch_size', 128, 'Batch size during training')
tf.flags.DEFINE_integer('epochs', 500, 'Epochs during training')
tf.flags.DEFINE_float('lr', 1.0e-3, 'Initial learing rate')
tf.flags.DEFINE_integer('embedding_size', 64, 'Embedding size for words')
tf.flags.DEFINE_integer('hidden_units', 256, 'Embedding size for words')
tf.flags.DEFINE_boolean('graph_write', True, 'whether the compute graph is written to logs file')
tf.flags.DEFINE_float('keep_prob', 0.9, 'The probility used to dropout')
tf.flags.DEFINE_string('mode', 'train0', 'The mode of train or predict as follows: '
                                          'train0: train first time or retrain'
                                          'train1: continue train'
                                          'predict: predict')
tf.flags.DEFINE_integer('per_save', 10, 'save model for every per_save')

FLAGS = tf.flags.FLAGS


def softmax(A, Mask):
    return tf.nn.softmax(tf.where(Mask, A, (1. - tf.pow(2., 31.)) * tf.ones_like(A)))


def create_rnn_cell(rnn_size, keep_prob):
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


def padding(x, l, padding_id):
    l_max = np.max(l)
    return [x[i] + [padding_id] * (l_max - l[i]) for i in range(len(x))]


def rearrange(x, r):
    return [x[ri] for ri in r]


class Seq2Seq():
    '''
    Seq2Seq模型
    go: start token
    eos: end token
    l_dict: number of words in dictionary
    '''

    def __init__(self, go=0, eos=1, l_dict=1000):
        self.go = go
        self.eos = eos
        self.l_dict = l_dict

    # 建立seq2seq的tensorflow模型
    def build_model(self):
        with tf.name_scope('Input'):
            encoder_inputs = tf.placeholder(shape=[None, None], dtype=tf.int32, name='encoder_inputs')
            decoder_inputs = tf.placeholder(shape=[None, None], dtype=tf.int32, name='decoder_inputs')
            decoder_targets = tf.placeholder(shape=[None, None], dtype=tf.int32, name='decoder_targets')

            encoder_length = tf.placeholder(shape=[None], dtype=tf.int32, name='encoder_length')
            decoder_length = tf.placeholder(shape=[None], dtype=tf.int32, name='decoder_length')
            max_decoder_length = tf.reduce_max(decoder_length, name='max_decoder_length')
            max_encoder_length = tf.reduce_max(encoder_length, name='max_encoder_length')

            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            mask_encoder = tf.sequence_mask(encoder_length, max_encoder_length)
            mask_decoder = tf.sequence_mask(decoder_length, max_decoder_length)

        with tf.name_scope('Embedding'):
            embedding_matrix = tf.Variable(
                tf.random_uniform([self.l_dict, FLAGS.embedding_size], -1.0, 1.0),
                dtype=tf.float32, name='embedding_matrix')
            encoder_inputs_embeded = tf.nn.embedding_lookup(embedding_matrix, encoder_inputs)
            decoder_inputs_embeded = tf.nn.embedding_lookup(embedding_matrix, decoder_inputs)

        with tf.variable_scope('encoder'):
            encoder_cell = create_rnn_cell(FLAGS.hidden_units, keep_prob)

            encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                                     inputs=encoder_inputs_embeded,
                                                                     sequence_length=encoder_length,
                                                                     dtype=tf.float32)

        with tf.variable_scope('decoder'):
            decoder_cell = create_rnn_cell(FLAGS.hidden_units, keep_prob)

            W_attn = tf.get_variable('W_attention', [2 * FLAGS.hidden_units, FLAGS.hidden_units])

            output_layer = tf.layers.Dense(self.l_dict, name='projection_layer')

            # for train
            if FLAGS.mode.startswith('train'):
                with tf.variable_scope('decoder'):
                    training_decoder_outputs = tf.zeros([0, FLAGS.hidden_units])

                    decoder_initial_state = encoder_final_state  # B*(2N)

                    i = tf.constant(0, tf.int32)

                    def cond_time(i, training_decoder_outputs, decoder_initial_state):
                        return tf.less(i, max_decoder_length)

                    def body_time(i, training_decoder_outputs, decoder_initial_state):
                        e_i = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(decoder_initial_state, W_attn), axis=1),
                                                   tf.transpose(encoder_outputs, [0, 2, 1])), axis=1)
                        weight = softmax(e_i, mask_encoder)

                        context = tf.squeeze(tf.matmul(tf.expand_dims(weight, axis=1), encoder_outputs), axis=1)

                        time_training_outputs, decoder_initial_state = decoder_cell(state=decoder_initial_state,
                                                                                    inputs=tf.concat([context,
                                                                                                      decoder_inputs_embeded[
                                                                                                      :, i]], axis=-1)
                                                                                    )
                        training_decoder_outputs = tf.concat([training_decoder_outputs, time_training_outputs], axis=0)
                        return tf.add(i, 1), training_decoder_outputs, decoder_initial_state

                    _, training_decoder_outputs, _ = tf.while_loop(cond_time, body_time,
                                                                   [i, training_decoder_outputs, decoder_initial_state],
                                                                   shape_invariants=[i.get_shape(),
                                                                                     tf.TensorShape([None,
                                                                                                     FLAGS.hidden_units]),
                                                                                     tf.TensorShape([None,
                                                                                                     2 * FLAGS.hidden_units])
                                                                                     ])
                    training_logits = tf.transpose(tf.reshape(output_layer(training_decoder_outputs),
                                                              [max_decoder_length, -1,
                                                               self.l_dict]), [1, 0, 2])


            # for infer
            else:
                with tf.variable_scope('decoder'):
                    prediction = tf.zeros([0], dtype=tf.int32)

                    decoder_initial_state = encoder_final_state

                    decoder_input = self.go * tf.ones_like(decoder_length, tf.int32)
                    i = tf.constant(0, tf.int32)

                    def cond_time_pre(i, prediction, decoder_initial_state, decoder_input):
                        return tf.less(i, max_decoder_length)

                    def body_time_pre(i, prediction, decoder_initial_state, decoder_input):
                        e_i = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(decoder_initial_state, W_attn), axis=1),
                                                   tf.transpose(encoder_outputs, [0, 2, 1])), axis=1)
                        weight = softmax(e_i, mask_encoder)

                        context = tf.squeeze(tf.matmul(tf.expand_dims(weight, axis=1), encoder_outputs), axis=1)

                        prediction_decoder_output, decoder_initial_state = decoder_cell(
                            tf.concat([context, tf.nn.embedding_lookup(embedding_matrix, decoder_input)], axis=-1),
                            decoder_initial_state)

                        prediction_output = output_layer(prediction_decoder_output)
                        decoder_input = tf.argmax(prediction_output, 1, output_type=tf.int32)
                        decoder_input = tf.where(mask_decoder[:, i], decoder_input, tf.zeros_like(decoder_input))
                        prediction = tf.concat([prediction, decoder_input], axis=0)

                        return tf.add(i, 1), prediction, decoder_initial_state, decoder_input

                    _, prediction, _, _ = tf.while_loop(cond_time_pre, body_time_pre,
                                                        [i, prediction, decoder_initial_state,
                                                         decoder_input],
                                                        shape_invariants=[i.get_shape(),
                                                                          tf.TensorShape([None]),
                                                                          tf.TensorShape([None,
                                                                                          2 * FLAGS.hidden_units]),
                                                                          tf.TensorShape([None])]
                                                        )
                    prediction = tf.transpose(tf.reshape(prediction, [max_decoder_length, -1]))

        if FLAGS.mode.startswith('train'):
            with tf.name_scope('Loss'):
                accuracy = tf.cast(tf.equal(tf.argmax(training_logits, axis=-1, output_type=tf.int32), decoder_targets),
                                   tf.float32)
                accuracyf = tf.zeros_like(accuracy)
                accuracy = tf.where(mask_decoder, accuracy, accuracyf)
                accuracy = tf.reduce_sum(accuracy) / tf.cast(tf.reduce_sum(decoder_length), tf.float32)
                accuracy = tf.identity(accuracy, name='accuracy')

                cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_targets, logits=training_logits)
                costf = tf.zeros_like(cost)
                loss = tf.reduce_mean(tf.div(tf.reduce_sum(tf.where(mask_decoder, cost, costf), axis=-1),
                                             tf.cast(decoder_length, tf.float32)), name='loss')

                optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr, name='optimizer')
                train_op = optimizer.minimize(loss, name='train_op')

            if FLAGS.graph_write:
                writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph(), filename_suffix='chatbot_0')
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
            saver.save(sess, FLAGS.model_save_path + 'chatbot_0')
            sess.close()
            print('Model saved successfully!')

        else:
            with tf.name_scope('Loss'):
                prediction = tf.identity(prediction, name='prediction')

    def train(self, qtext2id, qtext2id_length, atext2id_input, atext2id_target, atext2id_length, qa_dict,
              qa_reverse_dict):
        sess = tf.Session()

        new_saver = tf.train.import_meta_graph(FLAGS.model_save_path + 'chatbot_0.meta')
        new_saver.restore(sess, FLAGS.model_save_path + 'chatbot_0')

        graph = tf.get_default_graph()

        encoder_inputs = graph.get_operation_by_name('Input/encoder_inputs').outputs[0]
        decoder_inputs = graph.get_operation_by_name('Input/decoder_inputs').outputs[0]
        decoder_targets = graph.get_operation_by_name('Input/decoder_targets').outputs[0]
        keep_prob = graph.get_operation_by_name('Input/keep_prob').outputs[0]

        encoder_length = graph.get_operation_by_name('Input/encoder_length').outputs[0]
        decoder_length = graph.get_operation_by_name('Input/decoder_length').outputs[0]

        accuracy = graph.get_tensor_by_name('Loss/accuracy:0')
        loss = graph.get_tensor_by_name('Loss/loss:0')
        train_op = graph.get_operation_by_name('Loss/train_op')

        m_samples = len(qtext2id)
        total_batch = m_samples // FLAGS.batch_size

        loss_ = []
        acc_ = []
        for epoch in range(1, FLAGS.epochs + 1):
            loss_epoch = 0.0
            acc_epoch = 0.0
            for batch in range(total_batch):
                x_input_batch = qtext2id[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size]
                y_input_batch = atext2id_input[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size]
                y_target_batch = atext2id_target[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size]

                x_batch_length = qtext2id_length[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size]
                y_batch_length = atext2id_length[batch * FLAGS.batch_size:(batch + 1) * FLAGS.batch_size]

                x_input_batch = padding(x_input_batch, x_batch_length, qa_dict['<PAD>'])
                y_input_batch = padding(y_input_batch, y_batch_length, qa_dict['<PAD>'])
                y_target_batch = padding(y_target_batch, y_batch_length, qa_dict['<PAD>'])

                feed_dict = {
                    encoder_inputs: x_input_batch,
                    decoder_inputs: y_input_batch,
                    decoder_targets: y_target_batch,
                    encoder_length: x_batch_length,
                    decoder_length: y_batch_length,
                    keep_prob: FLAGS.keep_prob
                }
                loss_batch, acc_batch, _ = sess.run([loss, accuracy, train_op], feed_dict=feed_dict)

                loss_epoch += loss_batch
                acc_epoch += acc_batch

                sys.stdout.write('\r>> %d/%d | %d/%d  loss:%.9f  acc:%.2f%%' % (
                    epoch, FLAGS.epochs, batch + 1, total_batch, loss_batch, 100.0 * acc_batch))
                sys.stdout.flush()

            loss_.append(loss_epoch / total_batch)
            acc_.append(acc_epoch / total_batch)

            sys.stdout.write(' | Loss:%.9f  Acc:%.2f%%\n' % (loss_[-1], 100.0 * acc_[-1]))
            sys.stdout.flush()

            r = np.random.permutation(m_samples)
            qtext2id = rearrange(qtext2id, r)
            atext2id_input = rearrange(atext2id_input, r)
            atext2id_target = rearrange(atext2id_target, r)
            qtext2id_length = rearrange(qtext2id_length, r)
            atext2id_length = rearrange(atext2id_length, r)

            if epoch % FLAGS.per_save == 0:
                new_saver.save(sess, FLAGS.model_save_path + 'chatbot_0')
                print('Model saved successfully!')

        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(111)
        ax1.plot(loss_, 'r', label='Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(acc_, 'b', label='Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.legend(loc='upper right')

        plt.title('chatbot_0')
        plt.savefig(FLAGS.model_save_path + 'chatbot_0_loss_acc.png', bbox_inches='tight')
        plt.close(fig)

    def predict(self, qa_dict, qa_reverse_dict):
        sess = tf.Session()
        tf.train.Saver().restore(sess, FLAGS.model_save_path + 'chatbot_0')
        graph = tf.get_default_graph()

        encoder_inputs = graph.get_operation_by_name('Input/encoder_inputs').outputs[0]
        keep_prob = graph.get_operation_by_name('Input/keep_prob').outputs[0]
        encoder_length = graph.get_operation_by_name('Input/encoder_length').outputs[0]
        decoder_length = graph.get_operation_by_name('Input/decoder_length').outputs[0]
        prediction = graph.get_tensor_by_name('Loss/prediction:0')

        test = ['你好', '吃饭了吗']
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
        test_encoder_length = test_len
        test_decoder_length = [10, 12]

        print(test_encoder_input)
        prediction_ = sess.run(prediction, feed_dict={encoder_inputs: test_encoder_input,
                                                      encoder_length: test_encoder_length,
                                                      decoder_length: test_decoder_length,
                                                      keep_prob: 1.0})

        for i_test in range(len(test)):
            tmp = []
            for idx in prediction_[i_test]:
                if idx == qa_dict['<EOS>']:
                    break
                tmp.append(qa_reverse_dict[idx])
            sys.stdout.write('问: %s\n' % (test[i_test]))
            sys.stdout.write('答: %s\n\n' % (''.join(tmp)))
        sys.stdout.write(
            '-------------------------------------------------------------------------------------------------\n')
        sys.stdout.flush()

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

            prediction_ = sess.run(prediction, feed_dict={encoder_inputs: test_encoder_input,
                                                          encoder_length: test_encoder_length,
                                                          decoder_length: test_decoder_length,
                                                          keep_prob: 1.0})

            tmp = []
            for idx in prediction_[0]:
                if idx == qa_dict['<EOS>']:
                    break
                tmp.append(qa_reverse_dict[idx])
            sys.stdout.write('\n')
            sys.stdout.write('问: %s\n' % (test))
            sys.stdout.write('答: %s\n\n' % (''.join(tmp)))
            sys.stdout.flush()


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

    myseq2seq = Seq2Seq(go=qa_dict['<GO>'], eos=qa_dict['<EOS>'],
                        l_dict=len(qa_dict)
                        )

    if FLAGS.mode == 'train0':  # train first time or retrain
        if not os.path.exists(FLAGS.model_save_path):
            os.makedirs(FLAGS.model_save_path)

        myseq2seq.build_model()

        qtext2id, atext2id_input, atext2id_target = load_train_data()
        qtext2id_length = [len(q) for q in qtext2id]
        atext2id_length = [len(a) for a in atext2id_input]

        myseq2seq.train(qtext2id, qtext2id_length,
                        atext2id_input, atext2id_target, atext2id_length,
                        qa_dict, qa_reverse_dict
                        )

    elif FLAGS.mode == 'train1':  # continue train
        qtext2id, atext2id_input, atext2id_target = load_train_data()
        qtext2id_length = [len(q) for q in qtext2id]
        atext2id_length = [len(a) for a in atext2id_input]

        myseq2seq.train(qtext2id, qtext2id_length,
                        atext2id_input, atext2id_target, atext2id_length,
                        qa_dict, qa_reverse_dict
                        )
    elif FLAGS.mode == 'predict':
        myseq2seq.build_model()
        myseq2seq.predict(qa_dict, qa_reverse_dict)


if __name__ == '__main__':
    tf.app.run()
