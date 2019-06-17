'''
    Text Correction with BERT-CRF
'''

import tensorflow as tf
import numpy as np
import os
import sys
import matplotlib.pylab as plt
from source.PAST.CRF.make_tfrecord import label_dict, vocab_file
from source.PyCorrect.ZBERT.bert import modeling
from source.PyCorrect.ZBERT.bert import tokenization

os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # 使用GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # 屏蔽警告信息

tf.flags.DEFINE_integer('maxword', 512, 'max length of any sentences')
tf.flags.DEFINE_integer('batch_size', 100, 'batch size for training')
tf.flags.DEFINE_integer('epochs', 20000000, 'number of iterations')
tf.flags.DEFINE_string('model_save_path', 'model/mybert/bert.ckpt', 'directory of model file saved')
tf.flags.DEFINE_bool('bert_training', False, 'whether the bert model is training')
tf.flags.DEFINE_float('lr', 0.001, 'learning rate for training')
tf.flags.DEFINE_float('keep_prob', 0.5, 'rate for dropout')
tf.flags.DEFINE_integer('per_save', 1000, 'save model once every per_save iterations')
tf.flags.DEFINE_string('mode', 'train0', 'The mode of train or predict as follows: '
                                         'train0: train first time or retrain'
                                         'train1: continue train'
                                         'predict: predict')

config = tf.flags.FLAGS


def single_example_parser(serialized_example):
    context_features = {
        'length': tf.FixedLenFeature([], tf.int64)
    }
    sequence_features = {
        'sen': tf.FixedLenSequenceFeature([], tf.int64),
        'lab': tf.FixedLenSequenceFeature([], tf.int64)}

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    length = context_parsed['length']

    sen = sequence_parsed['sen']
    label = sequence_parsed['lab']
    return sen, label, length


def batched_data(tfrecord_filename, single_example_parser, batch_size, padded_shapes, buffer_size=1000):
    dataset = tf.data.TFRecordDataset(tfrecord_filename) \
        .map(single_example_parser) \
        .shuffle(buffer_size) \
        .repeat() \
        .padded_batch(batch_size, padded_shapes=padded_shapes)

    return dataset.make_one_shot_iterator().get_next()


class BertCRF():
    def __init__(self):
        self.label_dict_len = len(label_dict)
        is_training = False

        if config.mode.startswith('train'):
            is_training = True

        with tf.name_scope('input'):
            self.sen = tf.placeholder(tf.int32, [None, None], name='sentences')
            self.label = tf.placeholder(tf.int32, [None, None], name='labels')
            self.length = tf.placeholder(tf.int32, [None], name='length')
            max_length = tf.reduce_max(self.length, name='max_length')
            sequence_mask = tf.sequence_mask(self.length, max_length)

        model = modeling.BertModel(
            config=modeling.BertConfig.from_json_file("chinese_L-12_H-768_A-12/bert_config.json"),
            is_training=is_training,
            input_ids=self.sen,
            input_mask=tf.cast(sequence_mask, tf.int32),
            token_type_ids=tf.zeros_like(self.sen),
            use_one_hot_embeddings=False  # 这里如果使用TPU 设置为True，速度会快些。使用CPU 或GPU 设置为False ，速度会快些。
        )
        now = model.get_sequence_output()

        with tf.variable_scope('project'):
            if is_training:
                logits = tf.nn.dropout(tf.layers.dense(now, self.label_dict_len), keep_prob=config.keep_prob)
            else:
                logits = tf.layers.dense(now, self.label_dict_len)

        with tf.variable_scope('loss'):
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(logits, self.label, self.length)
            viterbi_sequence, _ = tf.contrib.crf.crf_decode(logits, transition_params, self.length)

            self.loss = tf.reduce_mean(-log_likelihood, name='loss')

            self.prediction = tf.identity(viterbi_sequence, 'prediction')

            accuracy = tf.cast(tf.equal(viterbi_sequence, self.label), tf.float32)
            accuracyf = tf.zeros_like(accuracy)
            self.accuracy = tf.div(tf.reduce_sum(tf.where(sequence_mask, accuracy, accuracyf)),
                                   tf.cast(tf.reduce_sum(self.length), tf.float32), name='accuracy')

        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=config.lr, name='optimizer')
            if config.bert_training:
                self.train_op = optimizer.minimize(self.loss, name='train_op')
            else:
                var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='project')
                var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='loss')
                self.train_op = optimizer.minimize(self.loss, var_list=var_list1 + var_list2, name='train_op')

        tvars = tf.trainable_variables()

        if config.mode == 'train0':
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, 'chinese_L-12_H-768_A-12/bert_model.ckpt')
            tf.train.init_from_checkpoint('chinese_L-12_H-768_A-12/bert_model.ckpt', assignment_map)

            # tf.logging.info("**** Trainable Variables ****")
            # for var in tvars:
            #     init_string = ""
            #     if var.name in initialized_variable_names:
            #         init_string = ", *INIT_FROM_CKPT*"
            #     tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
            #                     init_string)

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(max_to_keep=2)
            self.saver.save(self.sess, config.model_save_path)
            print('Model saved successfully!')
        else:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(max_to_keep=2)
            self.saver.restore(self.sess, config.model_save_path)
            print('Model restored successfully!')

        # number_trainable_variables = 0
        # variable_names = [v.name for v in tf.trainable_variables()]
        # values = self.sess.run(variable_names)
        # for k, v in zip(variable_names, values):
        #     print("Variable: ", k)
        #     print("Shape: ", v.shape)
        #     number_trainable_variables += np.prod([s for s in v.shape])
        # print('Number of parameters: %d' % number_trainable_variables)

    def train(self):
        train_file = ['data/train.tfrecord']

        train_batch = batched_data(train_file, single_example_parser, config.batch_size,
                                   padded_shapes=([-1], [-1], []))

        # ------------------------------------------------------------------------------------------------------------------
        label_reverse_dict = {v: k for k, v in label_dict.items()}
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)  # token 处理器，主要作用就是 分字，将字转换成ID。vocab_file 字典文件路径

        sentences = [
            '今天的信服来之不易',
            '中国人全的状况',
            '享有充分的权',
            '这是长期以来人泪追求的理想',
            '这是长期以来人类追求的理像',
            '这是长期以来人泪追求的理像',
            '成像居民收入稳定增长',
            '这是常见的一种的情况',
            '小学生应该未老年人让座'
        ]

        m_samples = len(sentences)

        sent = []
        leng = []
        for sentence in sentences:
            sen2id = tokenizer.convert_tokens_to_ids(['[CLS]'] + [token for token in sentence] + ['[SEP]'])
            sent.append(sen2id)
            leng.append(len(sen2id))

        max_len = np.max(leng)
        for i in range(m_samples):
            if leng[i] < max_len:
                sent[i] += tokenizer.convert_tokens_to_ids(['[PAD]']) * (max_len - leng[i])

        feed_dict_test = {self.sen: sent,
                          self.length: leng,
                          }
        # ------------------------------------------------------------------------------------------------------------------
        loss_ = []
        acc_ = []
        for epoch in range(1, config.epochs + 1):
            train_batch_ = self.sess.run(train_batch)

            feed_dict = {self.sen: train_batch_[0],
                         self.label: train_batch_[1],
                         self.length: train_batch_[2],
                         }
            loss_batch, acc_batch, _ = self.sess.run([self.loss, self.accuracy, self.train_op], feed_dict=feed_dict)
            loss_.append(loss_batch)
            acc_.append(acc_batch)

            sys.stdout.write('\r>> %d/%d  | loss_batch: %f  acc_batch:%.2f%%' % (
                epoch, config.epochs, loss_batch, 100.0 * acc_batch))
            sys.stdout.flush()

            if epoch % config.per_save == 0:
                sys.stdout.write('  train_loss:%f  train_acc:%.2f%% \n' % (
                    np.mean(loss_[-config.per_save:]),
                    100.0 * np.mean(acc_[-config.per_save:])))
                sys.stdout.flush()

                self.saver.save(self.sess, config.model_save_path)
                print('model saved successfully!')

                prediction_ = self.sess.run(self.prediction, feed_dict=feed_dict_test)

                for i in range(m_samples):
                    tmp = []
                    for idx in prediction_[i]:
                        tmp.append(label_reverse_dict[idx])
                    sys.stdout.write('SEN: %s\n' % (sentences[i]))
                    sys.stdout.write('LAB: %s\n\n' % (' '.join(tmp[1:leng[i] - 1])))
                sys.stdout.flush()

        self.sess.close()

        fig = plt.figure(figsize=(10, 8))
        plt.plot(loss_)
        plt.savefig(config.model_save_path + 'loss.png')
        plt.close(fig)

    def predict(self, sentences):
        label_reverse_dict = {v: k for k, v in label_dict.items()}
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)  # token 处理器，主要作用就是 分字，将字转换成ID。vocab_file 字典文件路径

        m_samples = len(sentences)

        sent = []
        leng = []
        for sentence in sentences:
            sen2id = tokenizer.convert_tokens_to_ids(['[CLS]'] + [token for token in sentence] + ['[SEP]'])
            sent.append(sen2id)
            leng.append(len(sen2id))

        max_len = np.max(leng)
        for i in range(m_samples):
            if leng[i] < max_len:
                sent[i] += tokenizer.convert_tokens_to_ids(['[PAD]']) * (max_len - leng[i])

        feed_dict_test = {self.sen: sent,
                          self.length: leng,
                          }

        sess = tf.Session()

        loader = tf.train.import_meta_graph(self.config.model_save_path + '.meta')
        loader.restore(sess, self.config.model_save_path)

        graph = tf.get_default_graph()

        sen = graph.get_operation_by_name('input/sentences').outputs[0]
        length = graph.get_operation_by_name('input/length').outputs[0]
        pos = graph.get_operation_by_name('input/position').outputs[0]
        keep_prob = graph.get_operation_by_name('input/keep_prob').outputs[0]

        prediction = graph.get_tensor_by_name('loss/prediction:0')

        pos_enc = np.array(
            [[position / np.power(10000.0, 2.0 * (i // 2) / self.config.embedding_size) for i in
              range(self.config.embedding_size)]
             for position in range(self.config.maxword)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])

        feed_dict = {sen: sent,
                     length: leng,
                     pos: pos_enc,
                     keep_prob: 1.0
                     }
        prediction_ = sess.run(prediction, feed_dict=feed_dict)

        for i in range(m_samples):
            tmp = []
            for idx in prediction_[i]:
                tmp.append(label_reverse_dict[idx])
            sys.stdout.write('SEN: %s\n' % (sentences[i]))
            sys.stdout.write('LAB: %s\n\n' % (' '.join(tmp[:leng[i]])))
        sys.stdout.flush()

        sess.close()


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    bertcrf = BertCRF()
    if config.mode.startswith('train'):
        bertcrf.train()

    # elif CONFIG.mode == 'predict':
    #     sentences = [
    #         '中国人权的状况的',
    #         '享有充分的权',
    #         '是长期以来人类追求的离乡',
    #         '成像居民收入稳定增长'
    #     ]
    #     bertcrf.predict(sentences,  label_dict)


if __name__ == '__main__':
    tf.app.run()
