'''
    生成JAVA所需的模型
    基于字预测mask来检验错误（检测log阈值候选项）
    一层XLNET进过修改的模型 PRI+THRESH
    逻辑或------->精准率较高
'''
import tensorflow as tf
from tensorflow.python.framework import graph_util
from chardict import load_vocab
import numpy as np

tf.flags.DEFINE_integer('maxword', 512, 'max length of any sentences')
tf.flags.DEFINE_integer('block', 4, 'number of Encoder submodel')
tf.flags.DEFINE_integer('head', 4, 'number of multi_head attention')

tf.flags.DEFINE_integer('embedding_size', 128, 'embedding size for word embedding')

CONFIG = tf.flags.FLAGS

def layer_norm_compute(x, scale, bias, epsilon=1.0e-6):
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
    Af = tf.ones_like(A) * (1. - tf.pow(2., 31.))
    A = tf.where(Mask, A, Af)
    C = tf.nn.softmax(A, axis=-1)
    return C


def gelu(input_tensor):
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


char_dict = load_vocab('vocab.txt')
char_dict_len = len(char_dict)

sen = tf.placeholder(tf.int32, [None, None], name='sen')
pri = tf.placeholder(tf.int32, name='pri')  # 提取前pri个可能的预测项
thresh = tf.placeholder(tf.float32, name='thresh')  # 提取前N个可能的阈值

max_ls = tf.shape(sen)[1]

pos_enc = np.array(
    [[position / np.power(10000.0, 2.0 * (i // 2) / CONFIG.embedding_size) for i in
      range(CONFIG.embedding_size)]
     for position in range(CONFIG.maxword)])
pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
pos_full = tf.constant(pos_enc, dtype=tf.float32, name='pos')
pos = tf.tile(tf.expand_dims(pos_full[:max_ls], axis=0), [tf.shape(sen)[0], 1, 1])

padding_mask = tf.tile(tf.expand_dims(tf.greater(sen, 0), 1), [CONFIG.head, tf.shape(sen)[1], 1])

future_mask = tf.tile(
    tf.expand_dims(tf.sequence_mask(tf.range(0, limit=tf.shape(sen)[1]), tf.shape(sen)[1]), 0),
    [tf.shape(sen)[0], 1, 1])
past_mask = tf.transpose(future_mask, perm=[0, 2, 1])
mask = tf.tile(tf.logical_or(future_mask, past_mask), [CONFIG.head, 1, 1])
mask_final = tf.logical_and(mask, padding_mask)

embedding_matrx = tf.get_variable('embedding_matrix',
                                  shape=[char_dict_len, CONFIG.embedding_size],
                                  initializer=tf.truncated_normal_initializer(stddev=0.02),
                                  dtype=tf.float32)
embedded_sen = tf.nn.embedding_lookup(embedding_matrx, sen)
scale_emb_sen = tf.get_variable('scale_emb_sen/layer_norm',
                                [CONFIG.embedding_size],
                                initializer=tf.ones_initializer(),
                                dtype=tf.float32)
bias_emb_sen = tf.get_variable('bias_emb_sen/layer_norm',
                               [CONFIG.embedding_size],
                               initializer=tf.zeros_initializer(),
                               dtype=tf.float32)
scale_emb_pos = tf.get_variable('scale_emb_pos/layer_norm',
                                [CONFIG.embedding_size],
                                initializer=tf.ones_initializer(),
                                dtype=tf.float32)
bias_emb_pos = tf.get_variable('bias_emb_pos/layer_norm',
                               [CONFIG.embedding_size],
                               initializer=tf.zeros_initializer(),
                               dtype=tf.float32)
now1 = layer_norm_compute(embedded_sen + pos, scale_emb_sen, bias_emb_sen)
now2 = layer_norm_compute(pos, scale_emb_pos, bias_emb_pos)

with tf.variable_scope('decoder'):
    for block in range(CONFIG.block):
        with tf.variable_scope('mutualattention' + str(block)):
            WQ = tf.layers.Dense(CONFIG.embedding_size,
                                 use_bias=False,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='Q')
            WK = tf.layers.Dense(CONFIG.embedding_size,
                                 use_bias=False,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='K')
            WV = tf.layers.Dense(CONFIG.embedding_size,
                                 use_bias=False,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='V')
            WO = tf.layers.Dense(CONFIG.embedding_size,
                                 use_bias=False,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='O')
            scale_sa_pos = tf.get_variable('scale_sa_pos/layer_norm',
                                           [CONFIG.embedding_size],
                                           initializer=tf.ones_initializer(),
                                           dtype=tf.float32)
            bias_sa_pos = tf.get_variable('bias_sa_pos/layer_norm',
                                          [CONFIG.embedding_size],
                                          initializer=tf.zeros_initializer(),
                                          dtype=tf.float32)
            # ------------------------------------------------------------------------------------------------------
            Q = tf.concat(tf.split(WQ(now2), CONFIG.head, axis=-1), axis=0)
            K = tf.concat(tf.split(WK(now1), CONFIG.head, axis=-1), axis=0)
            V = tf.concat(tf.split(WV(now1), CONFIG.head, axis=-1), axis=0)
            QK = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(
                CONFIG.embedding_size / CONFIG.head)
            Z = WO(tf.concat(
                tf.split(tf.matmul(softmax(QK, mask_final), V), CONFIG.head,
                         axis=0), axis=-1))
            now2 = layer_norm_compute(now2 + Z, scale_sa_pos, bias_sa_pos)
        with tf.variable_scope('feedforward' + str(block)):
            ffrelu = tf.layers.Dense(4 * CONFIG.embedding_size,
                                     activation=gelu,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     name='ffrelu')
            ff = tf.layers.Dense(CONFIG.embedding_size,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='ff')
            scale_ff_pos = tf.get_variable('scale_ff_pos/layer_norm',
                                           [CONFIG.embedding_size],
                                           initializer=tf.ones_initializer(),
                                           dtype=tf.float32)
            bias_ff_pos = tf.get_variable('bias_ff_pos/layer_norm',
                                          [CONFIG.embedding_size],
                                          initializer=tf.zeros_initializer(),
                                          dtype=tf.float32)

            now2 = layer_norm_compute(ff(ffrelu(now2)) + now2, scale_ff_pos, bias_ff_pos)

now = now2
scale_pr = tf.get_variable('scale_pr/layer_norm',
                           [CONFIG.embedding_size],
                           initializer=tf.ones_initializer(),
                           dtype=tf.float32)
bias_pr = tf.get_variable('bias_pr/layer_norm',
                          [CONFIG.embedding_size],
                          initializer=tf.zeros_initializer(),
                          dtype=tf.float32)
now = layer_norm_compute(tf.layers.dense(now,
                                         CONFIG.embedding_size,
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         activation=gelu,
                                         name='project'),
                         scale_pr, bias_pr)
project_bias = tf.get_variable('project_bias/bias',
                               [char_dict_len],
                               initializer=tf.zeros_initializer(),
                               dtype=tf.float32)

logits = tf.matmul(tf.reshape(now, [-1, CONFIG.embedding_size]), embedding_matrx,
                   transpose_b=True) + project_bias
logits = tf.reshape(logits, [-1, max_ls, char_dict_len])

topk = tf.argsort(logits, axis=-1, direction='DESCENDING')[:, :, :pri]
err1 = tf.reduce_sum(tf.cast(tf.equal(topk, tf.tile(tf.expand_dims(sen, axis=2), [1, 1, pri])), tf.int32),
                     axis=-1)

log_probs = tf.nn.log_softmax(logits, axis=-1)
one_hot_labels = tf.one_hot(sen, depth=char_dict_len, dtype=tf.float32)
per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])  # 概率对数的相反数
err2 = tf.cast(tf.less_equal(per_example_loss, thresh), tf.int32)
err = tf.cast(tf.greater(err1 + err2, 0), tf.int32, name='err')

init_checkpoint = 'model/dlm3/dlm.ckpt'  # tensorflow模型保存目录
sess = tf.Session()
sess.run(tf.global_variables_initializer())
restore_saver = tf.train.Saver()
restore_saver.restore(sess, init_checkpoint)

const_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['err'])
with tf.gfile.FastGFile("model/dlmmodel.pb", mode='wb') as f:
    f.write(const_graph.SerializeToString())
