'''
    My own Language Model for text correction using the first two layers of bert
    PRI
    THRESH
'''

import tensorflow as tf
import numpy as np
import os
import sys
import jieba
from bert import modeling
from chardict import load_vocab
import re
from tensorflow.python.framework import graph_util

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽警告信息
bert_path = 'chinese_L-12_H-768_A-12/'

tf.flags.DEFINE_string('bert_config_file', os.path.join(bert_path, 'bert_config.json'),
                       'config json file corresponding to the pre-trained BERT model.')
tf.flags.DEFINE_string('model_save_path', 'model/bertelmo/', 'The path where model shall be saved')
tf.flags.DEFINE_string('checkpoint', 'bertelmo.ckpt', 'The path where model shall be saved')
tf.flags.DEFINE_integer('batch_size', 10, 'batch size for training')
tf.flags.DEFINE_integer('epochs', 200000000, 'number of iterations')
tf.flags.DEFINE_integer('warmup', 200000, 'number of iterations')
tf.flags.DEFINE_float('lr', 1.0e-5, 'learning rate for training')
tf.flags.DEFINE_integer('per_save', 10000, 'save model once every per_save iterations')
tf.flags.DEFINE_integer('pri', 10, 'topk choices')
tf.flags.DEFINE_string('mode', 'train1', 'The mode of train or predict as follows: '
                                         'train0: train first time or retrain'
                                         'train1: continue train'
                                         'detect: error detect'
                                         'makepb: make model file'
                       )

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
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias", "output_bias"])

    # train_layers = [
    #     # 'layer_10',
    #     'layer_11',
    #     'cls'
    # ]
    # tvars = [v for v in tf.trainable_variables() if any(layers in v.name for layers in train_layers)]
    tvars = tf.trainable_variables()

    grads = tf.gradients(loss, tvars)

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)

    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)], name='train_op')
    return train_op, tvars


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


class BertElmo():
    def __init__(self, config, char_dict):
        self.config = config
        self.char_dict = char_dict
        self.char_dict_len = len(char_dict)
        self.reverse_char_dict = {v: k for k, v in char_dict.items()}

    def build_model(self):
        bert_config = modeling.BertConfig.from_json_file(self.config.bert_config_file)  # 获取BERT模型的各个超参数

        sen = tf.placeholder(tf.int32, [None, None], 'sen')  # 数字化文本
        sensen = tf.concat([sen, sen], axis=0)
        ls = tf.placeholder(tf.int32, [None], 'ls')  # 文本(实际)长度,去掉一头一尾
        pri = tf.placeholder(tf.int32, name='pri')  # 提取前pri个可能的预测项

        padding_mask = tf.sequence_mask(ls)
        # BERT预训练模型主体
        model = modeling.BertModel(
            config=bert_config,
            is_training=self.config.mode.startswith('train'),
            input_ids=sensen,
            input_mask=tf.greater(sensen, 0),
            token_type_ids=tf.zeros_like(sensen),
            use_one_hot_embeddings=False,
            scope='bert'
        )

        sequence_output = model.get_sequence_output()
        with tf.variable_scope("cls/predictions"):
            # We apply one more non-linear transformation before the output layer.
            # This matrix is not used after pre-training.
            with tf.variable_scope("transform"):
                sequence_output = tf.layers.dense(
                    sequence_output,
                    units=bert_config.hidden_size,
                    activation=modeling.get_activation(bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(
                        bert_config.initializer_range))
                sequence_output = modeling.layer_norm(sequence_output)

            # The output weights are the same as the input embeddings, but there is
            # an output-only bias for each token.
            output_bias = tf.get_variable(
                "output_bias",
                shape=[bert_config.vocab_size],
                initializer=tf.zeros_initializer())

        init_checkpoint = 'chinese_L-12_H-768_A-12/bert_model.ckpt'  # Bert模型保存目录
        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        out1, out2 = tf.split(sequence_output, 2)
        sequence_output = tf.concat([out1[:, :-2], out2[:, 2:]], axis=-1)
        sequence_output = tf.layers.dense(
            sequence_output,
            units=bert_config.hidden_size,
            activation=modeling.get_activation(bert_config.hidden_act),
            kernel_initializer=modeling.create_initializer(
                bert_config.initializer_range))
        sequence_output = modeling.layer_norm(sequence_output)

        projects = tf.matmul(tf.reshape(sequence_output, [-1, bert_config.hidden_size]), model.get_embedding_table(),
                             transpose_b=True)
        projects = tf.nn.bias_add(projects, output_bias)
        logits = tf.reshape(projects, [tf.shape(sen)[0], -1, bert_config.vocab_size], name='logits')

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sen[:, 1:-1], logits=logits)
        lossf = tf.zeros_like(loss)
        loss = tf.div(tf.reduce_sum(tf.where(padding_mask, loss, lossf)), tf.cast(tf.reduce_sum(ls), tf.float32),
                      name='loss')

        predict = tf.argmax(logits, axis=-1, output_type=tf.int32)
        acc = tf.cast(tf.equal(predict, sen[:, 1:-1]), tf.float32)
        accf = tf.zeros_like(acc)
        acc = tf.div(tf.reduce_sum(tf.where(padding_mask, acc, accf)), tf.cast(tf.reduce_sum(ls), tf.float32),
                     name='acc')

        train_op, tvars = create_optimizer(loss, self.config.lr, self.config.epochs, self.config.warmup)

        topk = tf.argsort(logits, axis=-1, direction='DESCENDING')[:, :, :pri]
        err = tf.reduce_sum(
            tf.cast(tf.equal(topk, tf.tile(tf.expand_dims(sen[:, 1:-1], axis=2), [1, 1, pri])), tf.int32),
            axis=-1, name='err')

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        number_trainable_variables = 0
        variable_names = [v.name for v in tvars]
        values = sess.run(variable_names)
        for k, v in zip(variable_names, values):
            print("Variable: ", k)
            print("Shape: ", v.shape)
            number_trainable_variables += np.prod([s for s in v.shape])
        print('Number of parameters: %d' % number_trainable_variables)

        saver = tf.train.Saver(max_to_keep=1)
        saver.save(sess, self.config.model_save_path + self.config.checkpoint)
        sess.close()
        print('Model saved successfully!')

    def train(self, train_file):
        sess = tf.Session()

        new_saver = tf.train.import_meta_graph(self.config.model_save_path + self.config.checkpoint + '.meta')
        new_saver.restore(sess, self.config.model_save_path + self.config.checkpoint)

        train_batch = batched_data(train_file, single_example_parser, self.config.batch_size,
                                   padded_shapes=([-1], []))

        graph = tf.get_default_graph()
        sen = graph.get_operation_by_name('sen').outputs[0]
        ls = graph.get_operation_by_name('ls').outputs[0]
        pri = graph.get_operation_by_name('pri').outputs[0]

        acc = graph.get_tensor_by_name('acc:0')
        loss = graph.get_tensor_by_name('loss:0')
        train_op = graph.get_operation_by_name('train_op')
        err = graph.get_tensor_by_name('err:0')

        sentences = [
            '证见会要切实维护的光大中小股动的核发权益。',
            '深沪交易将当天涨跌幅度炒锅7%的股票都称为异动股。',
            '现任中央财经大学研究员，同时兼任苏州工业园区凌志软件股份有限公司独立董事、国建新能科技股份有限公司董事。',
            '公司董事会、监事会及董事、监事、高级管理人员保证粘度报告内容的真实、准确、完整，不存在虚假记载、误导性陈述或重大遗漏，并承担个别和连带的法律责任。',
            '公司负责人易峥、主管会计工作负责人杜烈康及会计机构负责人(会计主管人员) 贾海明声明：暴政年度报告中财务报告的真实、准确、完整。',
            '所有董事均已出席了审议本报告的董事会会议。',
            '本报告设计的发展战略及未来前瞻性陈述，不构成公司对投资者的实质承诺。',
            '投资者及相关人士均应对此保持足够的奉献认识，并且应当理解计划、预测与承诺之间的差异。',
            '公司在经营管理中可能面临的风险与对策举措已在本报告中第四节“经营情况讨论与分析”之“九、公司未来发展的展望”部分予以秒数。',
            '敬请广大投资者管住，并注意投资风险。',
            '公司经本次董事会审议通过的利润分配裕安为：以537, 600, 000为基数，向全体股东每10股派发现金红利4.80元（含税），送红股0股（含税），以资本公积金向全体股东每10股转增0股。',
            '同时按照国际会计准这与按照中国会计准则披露的财务报告中净利润和净资产差异情况。',
            '对公司根据《公开发行证券的公司信息披露解释性公告第1号——非经常性损益》定义界定的非经常性顺义项目，以及把《公开发行证券的公司信息披露解释性公告第1号——非经常性损益》中列举的非经常性损益项目界定为经常性损益的项目，应说明原因。',
            '公司报告期不存在将根据《公开发行证券的公司信息披露解释性公告第1号——非经常性损益》定义、列举的非经常性损益项目界定为经常性损益的项目的情形。',
            '公司是否需要遵守特俗行业的披露要求。',
            '1、公司主营业务几经营情况。',
            '公司是国内领先的互联网金融信息服务提供商，产品及服务付该产业链上下有的各层次参与主体，包括证券公司、公募基金、私募基金、银行、保险、政府、研究机构、上市公司等机构客户，以及广大个人投资者。',
            '公司主要业务是为各类机构客户提供软件产品和系统维护服务、金融数据服务、智能推广服务， 为个人投资者提供金融咨询和投资理财分析工具。',
            '同时，公司基于现有的业务、技术、用户、数据优势， 积极探索和开发基于人工智能、大数据、云计算等前沿技术的产品和应用，以期形成新的业务漠视和增长点。',
            '目前，公司已构建同花顺AI开放平台，可面向客户提供智能语音、自然语言处理、智能金融问答、知识图谱、智能投顾等多项AI产品及服务，为银行、证券、保险、基金、私募等行业提供智能化解决方案。',
            '报告期内，公司努力耿总和把握行业技术和发展动向，以客户需求为中心，加强研发创新投入，积极做好新产品的研发和技术储备工作，进一步丰富产品和服务内容，提升公司核心竞争力。',
            '同时，公司全力推进各项资源正和，探索全新业务场景，充分挖掘行业潜能，培育新的业务增长点。',
            '2、公司所处行业发展地情况。',
            '自2005年中国证券市场实施股权分支改革以来，我国证券市场进入快速发展时期，互联网金融信息服务行业迎来了高速发展的机遇。',
            '互联网金融信息服务业在我过属于新兴行业，经营模式和盈利模式正在不断发展中。',
            '随着我国资本市场的不断壮大，居民财富的稳定增长，金融产品逐渐丰富，投资者对金融欣喜的需求不断增加，我国互联网金融信息服务市场容量不断上升，行业具有了一定规模，形成了从数据获取、数据处理到信息智能加工整合等较为完整的产业链。',
            '影响行业发展的因素主要击中在互联网整体环境和金融证券市场两方面。',
            '在互联网领域，影响因素包括互联网基础色是建设现状、技术革新、产业政策及互联网用户数量、用户支付习惯、网络安全体系建设等；在金融证券市场方面，影响因素主要有投资者数量、市场监管环境、证券市场趋势和交投活跃程度等。',
            '（1）技术革新为行业发展提供游离支持。',
            '计算机和互联网技术在近年来得发展及应用日趋深入，尤其是人工智能，大数据和云计算等关键领域的技术日新月异，为互联网金融信息服务企业通过前沿科技获得持续竞争优势，保持市场份额提供了重要支持。',
            '（2）产业政策有助与技术创新和产业升级。',
            '在“十三五”期间，国家出台了多项孤立企业进行技术创新和产业升级的政策。《“十三五”国家战略性新兴产业发展规划》明确指出，信息技术核心产业应顺应网络化、智能化、融合化等发展趋势，加快推动信息技术关键领域创新技术研发与产业化，推动电子信息产业转型升级。',
            '作为信息技术业的核心产业之一，互联网金融信息服务业椰浆迎来新的发展机遇。',
            '众多行业外的优秀企业已通过各种方是进行了互联网金融信息服务业的产业布局，未来的行业将迎来百花齐放的局面。',
            '（3）网民规模和手机上网比例不断增加，为行业发展奠定了坚实基础。',
            '根据中国胡连网络信息中心（CNNIC）发布的《第42次中国互联网络发展状况统计报告》，截至2018年6月，我国网民规模为8.02亿，上半年新增网民2968万人，较2017年末增加3.8 %，互联网普及率达57.7 %。',
            '公司董事会、监事会及董事、监事、高级管理人员保证粘度报告内容的真实、准确、完整，不存在虚假记载、误导性陈述或重大遗漏，并承担个别和连带的法律责任。',
            '公司负责人易峥、主管会计工作负责人杜烈康及会计机构付责人(会计主管人员)贾海明声明：保证年度报告中财务报告的真实、准确、完整。',
            '所有董事军已出席了审议本报告的董事会会议。',
            '本报告设计的发展战略及未来前瞻性陈述，不构成公司对投资者的实质承诺。',
            '投资者即相关人士均应对此保持足够的风险认识，并且应当理解计划、预测与承诺之间的差异。',
            '公司在经营管理中可能面料的风险与对策举措已在本报告中第四节“经营情况讨论与分析”之“九、公司未来发展的展望”部分予以描述。',
            '25、预计负债。',
            '31、租赁。',
            '（2）重要会计估计变更。',
            '22、商誉。'
        ]

        m_samples = len(sentences)

        sent = []
        leng = []
        for sentence in sentences:
            sen2id = [self.char_dict['[CLS]']] + [
                self.char_dict[word] if word in self.char_dict.keys() else self.char_dict['[UNK]']
                for word in sentence] + [self.char_dict['[SEP]']]
            sent.append(sen2id)
            leng.append(len(sen2id) - 2)

        max_len = np.max(leng)
        for i in range(m_samples):
            if leng[i] < max_len:
                pad = [self.char_dict['[PAD]']] * (max_len - leng[i])
                sent[i] += pad

        feed_dict_test = {sen: sent,
                          pri: self.config.pri,
                          }

        # ------------------------------------------------------------------------------------------------------------------

        loss_ = []
        acc_ = []
        for epoch in range(1, self.config.epochs + 1):
            train_batch_ = sess.run(train_batch)

            feed_dict = {sen: train_batch_[0],
                         ls: train_batch_[1]
                         }
            # print([self.reverse_char_dict[l] for l in train_batch_[0][0]])
            loss_batch, acc_batch, _ = sess.run([loss, acc, train_op], feed_dict=feed_dict)
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

                new_saver.save(sess, self.config.model_save_path + self.config.checkpoint)
                print('model saved successfully!')

                err_ = sess.run(err, feed_dict=feed_dict_test)

                # 检错环节
                for i in range(m_samples):
                    sentence = sentences[i]
                    error = err_[i]
                    words = jieba.lcut(sentence)
                    kwords = len(words)
                    ktext = len(sentence)

                    # l1 = -1
                    # for k in range(kwords):
                    #     lw = len(words[k])
                    #     if lw == 1:
                    #         l1 += 1
                    #     else:
                    #         for kk in range(l1 + 1, l1 + lw + 1):
                    #             if error[kk] == 0:
                    #                 error[l1 + 1:l1 + lw + 1] = 0
                    #                 break
                    #         l1 += lw

                    sys.stdout.write('检错: ')
                    for l in range(ktext):
                        if error[l] == 0:
                            sys.stdout.write('\033[4;31m%s\033[0m' % sentence[l])
                        else:
                            sys.stdout.write(sentence[l])
                    sys.stdout.write('\n\n')
                    sys.stdout.flush()

        sess.close()

    def detect(self, sentences):
        sess = tf.Session()

        new_saver = tf.train.import_meta_graph(self.config.model_save_path + self.config.checkpoint + '.meta')
        new_saver.restore(sess, self.config.model_save_path + self.config.checkpoint)

        graph = tf.get_default_graph()
        sen = graph.get_operation_by_name('sen').outputs[0]
        pri = graph.get_operation_by_name('pri').outputs[0]
        err = graph.get_tensor_by_name('err:0')

        m_samples = len(sentences)

        sent = []
        leng = []
        for sentence in sentences:
            sen2id = [self.char_dict['[CLS]']] + [
                self.char_dict[word] if word in self.char_dict.keys() else self.char_dict['[UNK]']
                for word in sentence] + [self.char_dict['[SEP]']]
            sent.append(sen2id)
            leng.append(len(sen2id) - 2)

        max_len = np.max(leng)
        for i in range(m_samples):
            if leng[i] < max_len:
                pad = [self.char_dict['[PAD]']] * (max_len - leng[i])
                sent[i] += pad

        feed_dict = {
            sen: sent,
            pri: self.config.pri,
        }

        err_ = sess.run(err, feed_dict=feed_dict)

        # 检错环节
        for i in range(m_samples):
            sentence = sentences[i]
            error = err_[i]
            words = jieba.lcut(sentence)
            kwords = len(words)
            ktext = len(sentence)

            # l1 = -1
            # for k in range(kwords):
            #     lw = len(words[k])
            #     if lw == 1:
            #         l1 += 1
            #     else:
            #         for kk in range(l1 + 1, l1 + lw + 1):
            #             if error[kk] == 0:
            #                 error[l1 + 1:l1 + lw + 1] = 0
            #                 break
            #         l1 += lw

            sys.stdout.write('检错: ')
            for l in range(ktext):
                if error[l] == 0:
                    sys.stdout.write('【%s】' % sentence[l])
                else:
                    sys.stdout.write(sentence[l])
            sys.stdout.write('\n\n')
            sys.stdout.flush()

        sess.close()

    def makepb(self):
        sess = tf.Session()

        new_saver = tf.train.import_meta_graph(self.config.model_save_path + self.config.checkpoint + '.meta')
        new_saver.restore(sess, self.config.model_save_path + self.config.checkpoint)

        const_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['err'])
        with tf.gfile.FastGFile("model/bertelmo.pb", mode='wb') as f:
            f.write(const_graph.SerializeToString())


def main(unused_argv):
    char_dict = load_vocab(bert_path + 'vocab.txt')

    bertelmo = BertElmo(CONFIG, char_dict)
    if CONFIG.mode == 'train0':
        if not os.path.exists(CONFIG.model_save_path):
            os.makedirs(CONFIG.model_save_path)
        bertelmo.build_model()
        bertelmo.train([
            # 'data/train_news2016zh.tfrecord',
            # 'data/train_wiki.tfrecord',
            'data/train_ann.tfrecord',
            'data/train_finance.tfrecord',
            'data/train_finance1.tfrecord',
        ])
    elif CONFIG.mode == 'train1':
        bertelmo.train([
            # 'data/train_news2016zh.tfrecord',
            # 'data/train_wiki.tfrecord',
            'data/train_ann.tfrecord',
            'data/train_finance.tfrecord',
            'data/train_finance1.tfrecord',
        ])
    elif CONFIG.mode == 'detect':
        sentences = [
            '证见会要切实维护的光大中小股动的核发权益。',
            '深沪交易将当天涨跌幅度炒锅7%的股票都称为异动股。',
            '现任中央财经大学研究员，同时兼任苏州工业园区凌志软件股份有限公司独立董事、国建新能科技股份有限公司董事。',
            '公司董事会、监事会及董事、监事、高级管理人员保证粘度报告内容的真实、准确、完整，不存在虚假记载、误导性陈述或重大遗漏，并承担个别和连带的法律责任。',
            '公司负责人易峥、主管会计工作负责人杜烈康及会计机构负责人(会计主管人员) 贾海明声明：暴政年度报告中财务报告的真实、准确、完整。',
            '所有董事均已出席了审议本报告的董事会会议。',
            '本报告设计的发展战略及未来前瞻性陈述，不构成公司对投资者的实质承诺。',
            '投资者及相关人士均应对此保持足够的奉献认识，并且应当理解计划、预测与承诺之间的差异。',
            '公司在经营管理中可能面临的风险与对策举措已在本报告中第四节“经营情况讨论与分析”之“九、公司未来发展的展望”部分予以秒数。',
            '敬请广大投资者管住，并注意投资风险。',
            '公司经本次董事会审议通过的利润分配裕安为：以537, 600, 000为基数，向全体股东每10股派发现金红利4.80元（含税），送红股0股（含税），以资本公积金向全体股东每10股转增0股。',
            '同时按照国际会计准这与按照中国会计准则披露的财务报告中净利润和净资产差异情况。',
            '对公司根据《公开发行证券的公司信息披露解释性公告第1号——非经常性损益》定义界定的非经常性顺义项目，以及把《公开发行证券的公司信息披露解释性公告第1号——非经常性损益》中列举的非经常性损益项目界定为经常性损益的项目，应说明原因。',
            '公司报告期不存在将根据《公开发行证券的公司信息披露解释性公告第1号——非经常性损益》定义、列举的非经常性损益项目界定为经常性损益的项目的情形。',
            '公司是否需要遵守特俗行业的披露要求。',
            '1、公司主营业务几经营情况。',
            '公司是国内领先的互联网金融信息服务提供商，产品及服务付该产业链上下有的各层次参与主体，包括证券公司、公募基金、私募基金、银行、保险、政府、研究机构、上市公司等机构客户，以及广大个人投资者。',
            '公司主要业务是为各类机构客户提供软件产品和系统维护服务、金融数据服务、智能推广服务， 为个人投资者提供金融咨询和投资理财分析工具。',
            '同时，公司基于现有的业务、技术、用户、数据优势， 积极探索和开发基于人工智能、大数据、云计算等前沿技术的产品和应用，以期形成新的业务漠视和增长点。',
            '目前，公司已构建同花顺AI开放平台，可面向客户提供智能语音、自然语言处理、智能金融问答、知识图谱、智能投顾等多项AI产品及服务，为银行、证券、保险、基金、私募等行业提供智能化解决方案。',
            '报告期内，公司努力耿总和把握行业技术和发展动向，以客户需求为中心，加强研发创新投入，积极做好新产品的研发和技术储备工作，进一步丰富产品和服务内容，提升公司核心竞争力。',
            '同时，公司全力推进各项资源正和，探索全新业务场景，充分挖掘行业潜能，培育新的业务增长点。',
            '2、公司所处行业发展地情况。',
            '自2005年中国证券市场实施股权分支改革以来，我国证券市场进入快速发展时期，互联网金融信息服务行业迎来了高速发展的机遇。',
            '互联网金融信息服务业在我过属于新兴行业，经营模式和盈利模式正在不断发展中。',
            '随着我国资本市场的不断壮大，居民财富的稳定增长，金融产品逐渐丰富，投资者对金融欣喜的需求不断增加，我国互联网金融信息服务市场容量不断上升，行业具有了一定规模，形成了从数据获取、数据处理到信息智能加工整合等较为完整的产业链。',
            '影响行业发展的因素主要击中在互联网整体环境和金融证券市场两方面。',
            '在互联网领域，影响因素包括互联网基础色是建设现状、技术革新、产业政策及互联网用户数量、用户支付习惯、网络安全体系建设等；在金融证券市场方面，影响因素主要有投资者数量、市场监管环境、证券市场趋势和交投活跃程度等。',
            '（1）技术革新为行业发展提供游离支持。',
            '计算机和互联网技术在近年来得发展及应用日趋深入，尤其是人工智能，大数据和云计算等关键领域的技术日新月异，为互联网金融信息服务企业通过前沿科技获得持续竞争优势，保持市场份额提供了重要支持。',
            '（2）产业政策有助与技术创新和产业升级。',
            '在“十三五”期间，国家出台了多项孤立企业进行技术创新和产业升级的政策。《“十三五”国家战略性新兴产业发展规划》明确指出，信息技术核心产业应顺应网络化、智能化、融合化等发展趋势，加快推动信息技术关键领域创新技术研发与产业化，推动电子信息产业转型升级。',
            '作为信息技术业的核心产业之一，互联网金融信息服务业椰浆迎来新的发展机遇。',
            '众多行业外的优秀企业已通过各种方是进行了互联网金融信息服务业的产业布局，未来的行业将迎来百花齐放的局面。',
            '（3）网民规模和手机上网比例不断增加，为行业发展奠定了坚实基础。',
            '根据中国胡连网络信息中心（CNNIC）发布的《第42次中国互联网络发展状况统计报告》，截至2018年6月，我国网民规模为8.02亿，上半年新增网民2968万人，较2017年末增加3.8 %，互联网普及率达57.7 %。',
            '公司董事会、监事会及董事、监事、高级管理人员保证粘度报告内容的真实、准确、完整，不存在虚假记载、误导性陈述或重大遗漏，并承担个别和连带的法律责任。',
            '公司负责人易峥、主管会计工作负责人杜烈康及会计机构付责人(会计主管人员)贾海明声明：保证年度报告中财务报告的真实、准确、完整。',
            '所有董事军已出席了审议本报告的董事会会议。',
            '本报告设计的发展战略及未来前瞻性陈述，不构成公司对投资者的实质承诺。',
            '投资者即相关人士均应对此保持足够的风险认识，并且应当理解计划、预测与承诺之间的差异。',
            '公司在经营管理中可能面料的风险与对策举措已在本报告中第四节“经营情况讨论与分析”之“九、公司未来发展的展望”部分予以描述。',
            '25、预计负债。',
            '31、租赁。',
            '（2）重要会计估计变更。',
            '22、商誉。'
        ]
        bertelmo.detect(sentences)
    elif CONFIG.mode == 'makepb':
        bertelmo.makepb()


if __name__ == '__main__':
    tf.app.run()
