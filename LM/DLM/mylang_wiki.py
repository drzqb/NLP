import tensorflow as tf
import re
from tqdm import tqdm
from chardict import load_vocab

tf.flags.DEFINE_integer('maxword', 512, 'max length of any sentences')

CONFIG = tf.flags.FLAGS


class Lang():
    def __init__(self, config):
        self.config = config
        self.read_data()
        self.toid()

    def read_data(self):
        fc = open('data/zhwiki_2017_03.txt', 'w', encoding='utf-8')
        m_samples = 0
        with  open('data/zhwiki_2017_03.clean', 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = re.split('\t', line.strip())
                if len(line) > 1:
                    sentences = re.split('([。])', re.sub(' ', '', line[1]))
                    sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2])]
                    for sentence in sentences:
                        if not sentence.endswith('。'):
                            sentence += '。'
                        if len(sentence) > 1 and len(sentence) <= self.config.maxword:
                            fc.write(sentence + '\n')
                            m_samples += 1

        fc.close()
        print('样本总量共：%d 句' % m_samples)  # 7793320 句

    def toid(self):
        char_dict = load_vocab('vocab.txt')

        writer = tf.python_io.TFRecordWriter('data/train_wiki.tfrecord')
        max_len = 0
        with  open('data/zhwiki_2017_03.txt', 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip()

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
        print('\n')
        print('最大序列长度: {}'.format(max_len))


def main(unused_argv):
    lang = Lang(CONFIG)


if __name__ == '__main__':
    tf.app.run()
