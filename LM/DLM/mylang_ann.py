import tensorflow as tf
from tqdm import tqdm
from chardict import load_vocab
import re
tf.flags.DEFINE_integer('maxword', 512, 'max length of any sentences')

CONFIG = tf.flags.FLAGS


class Lang():
    def __init__(self, config):
        self.config = config
        self.read_data()

    def read_data(self):
        m_samples = 0
        char_dict = load_vocab('vocab.txt')
        zhPattern = re.compile(u'[\u4e00-\u9fa5]+')

        writer = tf.python_io.TFRecordWriter('data/train_ann.tfrecord')
        max_len = 0
        with open('data/ann.txt', 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip()
                if (len(line) == 0) or (not zhPattern.search(line)):
                    continue
                if not line.endswith('。'):
                    line += '。'
                if len(line) <= self.config.maxword:
                    sen2id = [
                        char_dict[character] if character in char_dict.keys() else char_dict['[UNK]']
                        for character in line]

                    lensen = len(line)
                    if lensen > max_len:
                        max_len = lensen

                    length_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[lensen]))

                    sen_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in
                                   sen2id]

                    seq_example = tf.train.SequenceExample(
                        context=tf.train.Features(feature={
                            'ls': length_feature
                        }),
                        feature_lists=tf.train.FeatureLists(feature_list={
                            'sen': tf.train.FeatureList(feature=sen_feature)
                        })
                    )

                    serialized = seq_example.SerializeToString()

                    writer.write(serialized)
                    m_samples += 1
        print('\n')
        print('最大序列长度: {}'.format(max_len))

        print('样本总量共：%d 句' % m_samples)  # 4832 句


def main(unused_argv):
    lang = Lang(CONFIG)


if __name__ == '__main__':
    tf.app.run()
