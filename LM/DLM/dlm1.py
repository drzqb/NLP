'''
    My own Language Model for char using XLNET
    sen and pos 共用feedforward网络
    weight decay
    learning rate warmup and linear decay
    decoder每一层与encoder每一层注意力
'''

import tensorflow as tf
import numpy as np
import os
import sys
import jieba
from chardict import load_vocab
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽警告信息

tf.flags.DEFINE_integer('maxword', 512, 'max length of any sentences')
tf.flags.DEFINE_integer('block', 4, 'number of Encoder submodel')
tf.flags.DEFINE_integer('head', 4, 'number of multi_head attention')
tf.flags.DEFINE_string('model_save_path', 'model/dlm1/', 'The path where model shall be saved')
tf.flags.DEFINE_integer('batch_size', 100, 'batch size for training')
tf.flags.DEFINE_integer('epochs', 20000000, 'number of iterations')
tf.flags.DEFINE_integer('embedding_size', 128, 'embedding size for word embedding')
tf.flags.DEFINE_float('lr', 1.0e-3, 'learning rate for training')
tf.flags.DEFINE_integer('pri', 100, 'topk choices')
tf.flags.DEFINE_float('thresh', 7.7, 'thresh choices')
tf.flags.DEFINE_float('keep_prob', 0.9, 'rate for dropout')
tf.flags.DEFINE_integer('per_save', 10000, 'save model once every per_save iterations')
tf.flags.DEFINE_string('mode', 'train0', 'The mode of train or predict as follows: '
                                         'train0: train first time or retrain'
                                         'train1: continue train'
                                         'detect: error detect')

CONFIG = tf.flags.FLAGS


def single_example_parser(serialized_example):
    context_features = {
        'ls': tf.FixedLenFeature([], tf.int64)
    }
    sequence_features = {
        'sen': tf.FixedLenSequenceFeature([], tf.int64)
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    ls = context_parsed['ls']

    sen = sequence_parsed['sen']
    return sen, ls


def batched_data(tfrecord_filename, single_example_parser, batch_size, padded_shapes, buffer_size=1000):
    dataset = tf.data.TFRecordDataset(tfrecord_filename) \
        .map(single_example_parser) \
        .shuffle(buffer_size) \
        .repeat() \
        .padded_batch(batch_size, padded_shapes=padded_shapes)

    return dataset.make_one_shot_iterator().get_next()


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

def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps):
    """Creates an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["layer_norm", "bias"])

    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)

    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)], name='train_op')
    return train_op


class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:  # or param.name.startswith("bertsim")  将这个判断条件加入到之前，就不会训练Bert的参数。
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                    tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                    tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                              tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name

class DLM():
    def __init__(self, config, char_dict):
        self.config = config
        self.char_dict = char_dict
        self.char_dict_len = len(char_dict)
        self.char_reverse_dict = {v: k for k, v in char_dict.items()}

    def build_model(self):
        sen = tf.placeholder(tf.int32, [None, None], name='sen')
        ls = tf.placeholder(tf.int32, [None], name='ls')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        pri = tf.placeholder(tf.int32, name='pri')  # 提取前pri个可能的预测项
        thresh = tf.placeholder(tf.float32, name='thresh')  # 提取前N个可能的阈值

        max_ls = tf.shape(sen)[1]
        pos_enc = np.array(
            [[position / np.power(10000.0, 2.0 * (i // 2) / self.config.embedding_size) for i in
              range(self.config.embedding_size)]
             for position in range(self.config.maxword)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
        pos_full = tf.constant(pos_enc, dtype=tf.float32, name='pos')
        pos = tf.tile(tf.expand_dims(pos_full[:max_ls], axis=0), [tf.shape(sen)[0], 1, 1])
        padding_mask = tf.tile(tf.expand_dims(tf.greater(sen, 0), 1),
                               [self.config.head, tf.shape(sen)[1], 1])
        sequence_mask = tf.sequence_mask(ls)

        future_mask = tf.tile(
            tf.expand_dims(tf.sequence_mask(tf.range(0, limit=tf.shape(sen)[1]), tf.shape(sen)[1]), 0),
            [tf.shape(sen)[0], 1, 1])
        past_mask = tf.transpose(future_mask, perm=[0, 2, 1])
        mask = tf.tile(tf.logical_or(future_mask, past_mask), [self.config.head, 1, 1])
        mask_final = tf.logical_and(mask, padding_mask)

        embedding_matrx = tf.get_variable('embedding_matrix',
                                          shape=[self.char_dict_len, self.config.embedding_size],
                                          initializer=tf.truncated_normal_initializer(stddev=0.02),
                                          dtype=tf.float32)
        embedded_sen = tf.nn.embedding_lookup(embedding_matrx, sen)
        scale_emb_sen = tf.get_variable('scale_emb_sen/layer_norm',
                                        [self.config.embedding_size],
                                        initializer=tf.ones_initializer(),
                                        dtype=tf.float32)
        bias_emb_sen = tf.get_variable('bias_emb_sen/layer_norm',
                                       [self.config.embedding_size],
                                       initializer=tf.zeros_initializer(),
                                       dtype=tf.float32)
        scale_emb_pos = tf.get_variable('scale_emb_pos/layer_norm',
                                        [self.config.embedding_size],
                                        initializer=tf.ones_initializer(),
                                        dtype=tf.float32)
        bias_emb_pos = tf.get_variable('bias_emb_pos/layer_norm',
                                       [self.config.embedding_size],
                                       initializer=tf.zeros_initializer(),
                                       dtype=tf.float32)
        now1 = tf.nn.dropout(layer_norm_compute(embedded_sen + pos, scale_emb_sen, bias_emb_sen), keep_prob)
        now2 = tf.nn.dropout(layer_norm_compute(pos, scale_emb_pos, bias_emb_pos), keep_prob)

        for block in range(self.config.block):
            with tf.variable_scope('mutualattention' + str(block)):
                WQ = tf.layers.Dense(self.config.embedding_size,
                                     use_bias=False,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     name='Q')
                WK = tf.layers.Dense(self.config.embedding_size,
                                     use_bias=False,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     name='K')
                WV = tf.layers.Dense(self.config.embedding_size,
                                     use_bias=False,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     name='V')
                WO = tf.layers.Dense(self.config.embedding_size,
                                     use_bias=False,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     name='O')
                scale_sa_pos = tf.get_variable('scale_sa_pos/layer_norm',
                                               [self.config.embedding_size],
                                               initializer=tf.ones_initializer(),
                                               dtype=tf.float32)
                bias_sa_pos = tf.get_variable('bias_sa_pos/layer_norm',
                                              [self.config.embedding_size],
                                              initializer=tf.zeros_initializer(),
                                              dtype=tf.float32)
                # ------------------------------------------------------------------------------------------------------
                Q = tf.concat(tf.split(WQ(now2), self.config.head, axis=-1), axis=0)
                K = tf.concat(tf.split(WK(now1), self.config.head, axis=-1), axis=0)
                V = tf.concat(tf.split(WV(now1), self.config.head, axis=-1), axis=0)
                QK = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(self.config.embedding_size / self.config.head)
                Z = WO(tf.concat(
                    tf.split(tf.matmul(tf.nn.dropout(softmax(QK, mask_final), keep_prob), V), self.config.head,
                             axis=0), axis=-1))
                now2 = tf.nn.dropout(layer_norm_compute(now2 + Z, scale_sa_pos, bias_sa_pos), keep_prob)
                now = tf.concat([now1, now2], axis=0)
            with tf.variable_scope('feedforward' + str(block)):
                ffrelu = tf.layers.Dense(4 * self.config.embedding_size,
                                         activation=gelu,
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         name='ffrelu')
                ff = tf.layers.Dense(self.config.embedding_size,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     name='ff')
                scale_ff_sen = tf.get_variable('scale_ff_sen/layer_norm',
                                               [self.config.embedding_size],
                                               initializer=tf.ones_initializer(),
                                               dtype=tf.float32)
                bias_ff_sen = tf.get_variable('bias_ff_sen/layer_norm',
                                              [self.config.embedding_size],
                                              initializer=tf.zeros_initializer(),
                                              dtype=tf.float32)

                scale_ff_pos = tf.get_variable('scale_ff_pos/layer_norm',
                                               [self.config.embedding_size],
                                               initializer=tf.ones_initializer(),
                                               dtype=tf.float32)
                bias_ff_pos = tf.get_variable('bias_ff_pos/layer_norm',
                                              [self.config.embedding_size],
                                              initializer=tf.zeros_initializer(),
                                              dtype=tf.float32)

                now1, now2 = tf.split(ff(ffrelu(now)) + now, 2)

                now1 = tf.nn.dropout(layer_norm_compute(now1, scale_ff_sen, bias_ff_sen), keep_prob)
                now2 = tf.nn.dropout(layer_norm_compute(now2, scale_ff_pos, bias_ff_pos), keep_prob)

        now = now2
        scale_pr = tf.get_variable('scale_pr/layer_norm',
                                   [self.config.embedding_size],
                                   initializer=tf.ones_initializer(),
                                   dtype=tf.float32)
        bias_pr = tf.get_variable('bias_pr/layer_norm',
                                  [self.config.embedding_size],
                                  initializer=tf.zeros_initializer(),
                                  dtype=tf.float32)
        now = layer_norm_compute(tf.layers.dense(now,
                                                 self.config.embedding_size,
                                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                 activation=gelu,
                                                 name='project'),
                                 scale_pr, bias_pr)
        project_bias = tf.get_variable('project_bias/bias',
                                       [self.char_dict_len],
                                       initializer=tf.zeros_initializer(),
                                       dtype=tf.float32)

        logits = tf.matmul(tf.reshape(now, [-1, self.config.embedding_size]), embedding_matrx,
                           transpose_b=True) + project_bias
        logits = tf.reshape(logits, [-1, max_ls, self.char_dict_len])

        prediction = tf.argmax(logits, axis=-1, output_type=tf.int32)
        predictionf = tf.zeros_like(prediction)
        prediction = tf.where(sequence_mask, prediction, predictionf)

        accuracy = tf.cast(tf.equal(prediction, sen), tf.float32)
        accuracyf = tf.zeros_like(accuracy)
        accuracy = tf.div(tf.reduce_sum(tf.where(sequence_mask, accuracy, accuracyf)),
                          tf.cast(tf.reduce_sum(ls), tf.float32), name='accuracy')

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sen, logits=logits)
        lossf = tf.zeros_like(loss)
        loss = tf.div(tf.reduce_sum(tf.where(sequence_mask, loss, lossf)),
                      tf.cast(tf.reduce_sum(ls), tf.float32), name='loss')

        train_op = create_optimizer(loss, self.config.lr, self.config.epochs, 10000)

        topk = tf.argsort(logits, axis=-1, direction='DESCENDING')[:, :, :pri]
        err1 = tf.reduce_sum(tf.cast(tf.equal(topk, tf.tile(tf.expand_dims(sen, axis=2), [1, 1, pri])), tf.int32),
                             axis=-1)

        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(sen, depth=self.char_dict_len, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])  # 概率对数的相反数
        err2 = tf.cast(tf.less_equal(per_example_loss, thresh), tf.int32)
        err = tf.cast(tf.greater(err1 + err2, 0), tf.int32, name='err')

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
        saver.save(sess, self.config.model_save_path + 'dlm.ckpt')
        sess.close()
        print('Model saved successfully!')

    def train(self, train_file):
        sess = tf.Session()

        new_saver = tf.train.import_meta_graph(self.config.model_save_path + 'dlm.ckpt.meta')
        new_saver.restore(sess, self.config.model_save_path + 'dlm.ckpt')

        train_batch = batched_data(train_file, single_example_parser, self.config.batch_size,
                                   padded_shapes=([-1], []))

        graph = tf.get_default_graph()
        sen = graph.get_operation_by_name('sen').outputs[0]
        ls = graph.get_operation_by_name('ls').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
        pri = graph.get_operation_by_name('pri').outputs[0]
        thresh = graph.get_operation_by_name('thresh').outputs[0]

        err = graph.get_tensor_by_name('err:0')
        accuracy = graph.get_tensor_by_name('accuracy:0')
        loss = graph.get_tensor_by_name('loss:0')
        train_op = graph.get_operation_by_name('train_op')

        sentences = [
            '公司董事会、监事会及董事、监事、高级管理人员保证粘度报告内容的真实、准确、完整，不存在虚假记载、误导性陈述或重大遗漏，并承担个别和连带的法律责任。',
            '公司负责人易峥、主管会计工作负责人杜烈康及会计机构负责人(会计主管人员) 贾海明声明：暴政年度报告中财务报告的真实、准确、完整。',
            '投资者及相关人士均应对此保持足够的奉献认识，并且应当理解计划、预测与承诺之间的差异。',
            '敬请广大投资者管住，并注意投资风险。',
            '公司经本次董事会审议通过的利润分配裕安为：以537, 600, 000为基数，向全体股东每10股派发现金红利4.80元（含税），送红股0股（含税），以资本公积金向全体股东每10股转增0股。',
            '对公司根据《公开发行证券的公司信息披露解释性公告第1号——非经常性损益》定义界定的非经常性顺义项目，以及把《公开发行证券的公司信息披露解释性公告第1号——非经常性损益》中列举的非经常性损益项目界定为经常性损益的项目，应说明原因。',
            '公司报告期不存在将根据《公开发行证券的公司信息披露解释性公告第1号——非经常性损益》定义、列举的非经常性损益项目界定为经常性损益的项目的情形。',
            '公司是否需要遵守特俗行业的披露要求 否。',
            '1、公司主营业务几经营情况。',
            '公司是国内领先的互联网金融信息服务提供商，产品及服务付该产业链上下有的各层次参与主体，包括证券公司、公募基金、私募基金、银行、保险、政府、研究机构、上市公司等机构客户，以及广大个人投资者。',
            '目前，公司已构建同花顺AI开放平台，可面向客户提供智能语音、自然语言处理、智能金融问答、知识图谱、智能投顾等多项AI产品及服务，为银行、证券、保险、基金、私募等行业提供智能化解决方案。',
            '同时，公司全力推进各项资源正和，探索全新业务场景，充分挖掘行业潜能，培育新的业务增长点。',
            '2、公司所处行业发展情况。',
            '在互联网领域，影响因素包括互联网基础设施建设现状、技术革新、产业政策及互联网用户数量、用户支付习惯、网络安全体系建设等；在金融证券市场方面，影响因素主要有投资者数量、市场监管环境、证券市场趋势和交投活跃程度等。',
            '（1）技术革新为行业发展提供有力支持。',
            '计算机和互联网技术在近年来的发展及应用日趋深入，尤其是人工智能，大数据和云计算等关键领域的技术日新月异，为互联网金融信息服务企业通过前沿科技获得持续竞争优势，保持市场份额提供了重要支持。',
            '（2）产业政策有助于技术创新和产业升级。',
            '在“十三五”期间，国家出台了多项鼓励企业进行技术创新和产业升级的政策。《“十三五”国家战略性新兴产业发展规划》明确指出，信息技术核心产业应顺应网络化、智能化、融合化等发展趋势，加快推动信息技术关键领域创新技术研发与产业化，推动电子信息产业转型升级。',
            '作为信息技术业的核心产业之一，互联网金融信息服务业也将迎来新的发展机遇。',
            '众多行业外的优秀企业已通过各种方式进行了互联网金融信息服务业的产业布局，未来的行业将迎来百花齐放的局面。',
            '（3）网民规模和手机上网比例不断增加，为行业发展奠定了坚实基础。',
            '根据中国胡连网络信息中心（CNNIC）发布的《第42次中国互联网络发展状况统计报告》，截至2018年6月，我国网民规模为8.02亿，上半年新增网民2968万人，较2017年末增加3.8 %，互联网普及率达57.7 %。',
            '公司董事会、监事会及董事、监事、高级管理人员保证粘度报告内容的真实、准确、完整，不存在虚假记载、误导性陈述或重大遗漏，并承担个别和连带的法律责任。',
            '公司在经营管理中可能面临的风险与对策举措已在本报告中第四节“经营情况讨论与分析”之“九、公司未来发展的展望”部分予以描述。',
            '25、预计负债。',
            '31、租赁。',
            '（2）重要会计估计变更。',
            '22、商誉。'
        ]

        m_samples = len(sentences)

        sent = []
        leng = []
        for sentence in sentences:
            sen2id = [
                self.char_dict[word] if word in self.char_dict.keys() else self.char_dict['[UNK]']
                for word in sentence]
            sent.append(sen2id)
            leng.append(len(sentence))

        max_len = np.max(leng)
        for i in range(m_samples):
            if leng[i] < max_len:
                pad = [self.char_dict['[PAD]']] * (max_len - leng[i])
                sent[i] += pad

        feed_dict_test = {sen: sent,
                          ls: leng,
                          keep_prob: 1.0,
                          pri: self.config.pri,
                          thresh: self.config.thresh
                          }

        # ------------------------------------------------------------------------------------------------------------------

        loss_ = []
        acc_ = []
        for epoch in range(1, self.config.epochs + 1):
            train_batch_ = sess.run(train_batch)

            feed_dict = {sen: train_batch_[0],
                         ls: train_batch_[1],
                         keep_prob: self.config.keep_prob
                         }
            loss_batch, acc_batch, _ = sess.run([loss, accuracy, train_op], feed_dict=feed_dict)
            loss_.append(loss_batch)
            acc_.append(acc_batch)

            sys.stdout.write('\r>> %d/%d  | loss_batch: %f  acc_batch:%.3f%%' % (
                epoch, self.config.epochs, loss_batch, 100.0 * acc_batch))
            sys.stdout.flush()

            if epoch % self.config.per_save == 0:
                sys.stdout.write('  train_loss:%f  train_acc:%.3f%%\n' % (
                    np.mean(loss_[-self.config.per_save:]),
                    100.0 * np.mean(acc_[-self.config.per_save:])))
                sys.stdout.flush()

                new_saver.save(sess, self.config.model_save_path + 'dlm.ckpt')
                print('model saved successfully!')

                err_ = sess.run(err, feed_dict=feed_dict_test)

                # 检错环节
                for i in range(m_samples):
                    sentence = sentences[i]
                    error = err_[i]
                    words = jieba.lcut(sentence)
                    kwords = len(words)

                    l1 = -1
                    for k in range(kwords):
                        lw = len(words[k])
                        if lw == 1:
                            l1 += 1
                        else:
                            for kk in range(l1 + 1, l1 + lw + 1):
                                if error[kk] == 0:
                                    error[l1 + 1:l1 + lw + 1] = 0
                                    break
                            l1 += lw

                    sys.stdout.write('检错: ')
                    l1 = 0
                    for l in range(kwords):
                        if error[l1] == 0:
                            sys.stdout.write('【%s】' % words[l])
                        else:
                            sys.stdout.write(words[l])
                        l1 += len(words[l])
                    sys.stdout.write('\n\n')
                    sys.stdout.flush()

        sess.close()

    def detect(self, sentences):
        sess = tf.Session()

        new_saver = tf.train.import_meta_graph(self.config.model_save_path + 'dlm.ckpt.meta')
        new_saver.restore(sess, self.config.model_save_path + 'dlm.ckpt')

        graph = tf.get_default_graph()
        sen = graph.get_operation_by_name('sen').outputs[0]
        ls = graph.get_operation_by_name('ls').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
        pri = graph.get_operation_by_name('pri').outputs[0]
        thresh = graph.get_operation_by_name('thresh').outputs[0]

        err = graph.get_tensor_by_name('err:0')

        m_samples = len(sentences)

        sent = []
        leng = []
        for sentence in sentences:
            sen2id = [
                self.char_dict[word] if word in self.char_dict.keys() else self.char_dict['[UNK]']
                for word in sentence]
            sent.append(sen2id)
            leng.append(len(sentence))

        max_len = np.max(leng)
        for i in range(m_samples):
            if leng[i] < max_len:
                pad = [self.char_dict['[PAD]']] * (max_len - leng[i])
                sent[i] += pad

        feed_dict = {
            sen: sent,
            ls: leng,
            keep_prob: 1.0,
            pri: self.config.pri,
            thresh: self.config.thresh
        }

        err_ = sess.run(err, feed_dict=feed_dict)
        # 检错环节
        for i in range(m_samples):
            sentence = sentences[i]
            error = err_[i]
            words = jieba.lcut(sentence)
            kwords = len(words)

            l1 = -1
            for k in range(kwords):
                lw = len(words[k])
                if lw == 1:
                    l1 += 1
                else:
                    for kk in range(l1 + 1, l1 + lw + 1):
                        if error[kk] == 0:
                            error[l1 + 1:l1 + lw + 1] = 0
                            break
                    l1 += lw

            sys.stdout.write('检错: ')
            l1 = 0
            for l in range(kwords):
                if error[l1] == 0:
                    sys.stdout.write('【%s】' % words[l])
                else:
                    sys.stdout.write(words[l])
                l1 += len(words[l])
            sys.stdout.write('\n\n')
            sys.stdout.flush()


def main(unused_argv):
    char_dict = load_vocab('vocab.txt')

    dlm = DLM(CONFIG, char_dict)
    if CONFIG.mode == 'train0':
        if not os.path.exists(CONFIG.model_save_path):
            os.makedirs(CONFIG.model_save_path)
        dlm.build_model()
        dlm.train([
            'data/train_wiki.tfrecord',
            'data/train_ann.tfrecord',
            'data/train_finance.tfrecord'
        ])
    elif CONFIG.mode == 'train1':
        dlm.train([
            'data/train_wiki.tfrecord',
            'data/train_ann.tfrecord',
            'data/train_finance.tfrecord'
        ])
    elif CONFIG.mode == 'detect':
        sentences = [
            '公司董事会、监事会及董事、监事、高级管理人员保证粘度报告内容的真实、准确、完整，不存在虚假记载、误导性陈述或重大遗漏，并承担个别和连带的法律责任。',
            '公司负责人易峥、主管会计工作负责人杜烈康及会计机构负责人(会计主管人员) 贾海明声明：暴政年度报告中财务报告的真实、准确、完整。',
            '投资者及相关人士均应对此保持足够的奉献认识，并且应当理解计划、预测与承诺之间的差异。',
            '敬请广大投资者管住，并注意投资风险。',
            '公司经本次董事会审议通过的利润分配裕安为：以537, 600, 000为基数，向全体股东每10股派发现金红利4.80元（含税），送红股0股（含税），以资本公积金向全体股东每10股转增0股。',
            '对公司根据《公开发行证券的公司信息披露解释性公告第1号——非经常性损益》定义界定的非经常性顺义项目，以及把《公开发行证券的公司信息披露解释性公告第1号——非经常性损益》中列举的非经常性损益项目界定为经常性损益的项目，应说明原因。',
            '公司报告期不存在将根据《公开发行证券的公司信息披露解释性公告第1号——非经常性损益》定义、列举的非经常性损益项目界定为经常性损益的项目的情形。',
            '公司是否需要遵守特俗行业的披露要求 否。',
            '1、公司主营业务几经营情况。',
            '公司是国内领先的互联网金融信息服务提供商，产品及服务付该产业链上下有的各层次参与主体，包括证券公司、公募基金、私募基金、银行、保险、政府、研究机构、上市公司等机构客户，以及广大个人投资者。',
            '目前，公司已构建同花顺AI开放平台，可面向客户提供智能语音、自然语言处理、智能金融问答、知识图谱、智能投顾等多项AI产品及服务，为银行、证券、保险、基金、私募等行业提供智能化解决方案。',
            '同时，公司全力推进各项资源正和，探索全新业务场景，充分挖掘行业潜能，培育新的业务增长点。',
            '2、公司所处行业发展情况。',
            '在互联网领域，影响因素包括互联网基础设施建设现状、技术革新、产业政策及互联网用户数量、用户支付习惯、网络安全体系建设等；在金融证券市场方面，影响因素主要有投资者数量、市场监管环境、证券市场趋势和交投活跃程度等。',
            '（1）技术革新为行业发展提供有力支持。',
            '计算机和互联网技术在近年来的发展及应用日趋深入，尤其是人工智能，大数据和云计算等关键领域的技术日新月异，为互联网金融信息服务企业通过前沿科技获得持续竞争优势，保持市场份额提供了重要支持。',
            '（2）产业政策有助于技术创新和产业升级。',
            '在“十三五”期间，国家出台了多项鼓励企业进行技术创新和产业升级的政策。《“十三五”国家战略性新兴产业发展规划》明确指出，信息技术核心产业应顺应网络化、智能化、融合化等发展趋势，加快推动信息技术关键领域创新技术研发与产业化，推动电子信息产业转型升级。',
            '作为信息技术业的核心产业之一，互联网金融信息服务业也将迎来新的发展机遇。',
            '众多行业外的优秀企业已通过各种方式进行了互联网金融信息服务业的产业布局，未来的行业将迎来百花齐放的局面。',
            '（3）网民规模和手机上网比例不断增加，为行业发展奠定了坚实基础。',
            '根据中国胡连网络信息中心（CNNIC）发布的《第42次中国互联网络发展状况统计报告》，截至2018年6月，我国网民规模为8.02亿，上半年新增网民2968万人，较2017年末增加3.8 %，互联网普及率达57.7 %。',
            '公司董事会、监事会及董事、监事、高级管理人员保证粘度报告内容的真实、准确、完整，不存在虚假记载、误导性陈述或重大遗漏，并承担个别和连带的法律责任。',
            '公司在经营管理中可能面临的风险与对策举措已在本报告中第四节“经营情况讨论与分析”之“九、公司未来发展的展望”部分予以描述。',
            '25、预计负债。',
            '31、租赁。',
            '（2）重要会计估计变更。',
            '22、商誉。'
        ]
        dlm.detect(sentences)


if __name__ == '__main__':
    tf.app.run()
