import tensorflow as tf
import pickle
import re
from tqdm import tqdm

tf.flags.DEFINE_integer('most_common', 50000, 'number of most common used words')
tf.flags.DEFINE_integer('maxword', 510, 'max length of any sentences')

CONFIG = tf.flags.FLAGS


class Lang():
    def __init__(self, config):
        self.config = config
        self.read_data()
        self.toid()

    def read_data(self):
        character_dict = dict()
        fc = open('../data/zhwiki_2017_03.txt', 'w', encoding='utf-8')
        m_samples = 0
        with  open('../data/zhwiki_2017_03.clean', 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = re.split('\t', line.strip())
                if len(line) > 1:
                    sentences = re.split('([。])', re.sub(' ', '', line[1]))
                    sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2])]
                    for sentence in sentences:
                        if len(sentence) > 5 and len(sentence) <= self.config.maxword:
                            fc.write(sentence + '\n')
                            m_samples += 1
                            for word in sentence:
                                if word in character_dict.keys():
                                    character_dict[word] += 1
                                else:
                                    character_dict[word] = 1
                        elif len(sentence) > self.config.maxword:
                            sentence = re.split('，', sentence)
                            for sen in sentence:
                                if len(sen) > 5 and len(sen) <= self.config.maxword:
                                    fc.write(sen + '\n')
                                    m_samples += 1
                                    for word in sen:
                                        if word in character_dict.keys():
                                            character_dict[word] += 1
                                        else:
                                            character_dict[word] = 1

        fc.close()
        print('样本总量共：%d 句' % m_samples)

        char_dict = dict()
        char_dict['<pad>'] = 0
        char_dict['<go>'] = 1
        char_dict['<eos>'] = 2
        char_dict['<unknown>'] = 3

        dd = sorted(character_dict.items(), key=lambda x: x[1], reverse=True)
        k = 4
        for d in dd:
            if k == self.config.most_common:
                break
            char_dict[d[0]] = k
            k = k + 1

        with open('../data/char_dict.txt', 'wb') as f:
            pickle.dump(char_dict, f)

        print('字典大小：%d' % (len(char_dict)))

    def toid(self):
        with open('../data/char_dict.txt', 'rb') as f:
            char_dict = pickle.load(f)

        writer = tf.python_io.TFRecordWriter('../data/train.tfrecord')
        max_len = 0
        with  open('../data/zhwiki_2017_03.txt', 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip()

                sen2id = [char_dict['<go>']] + [
                    char_dict[character] if character in char_dict.keys() else char_dict['<unknown>']
                    for character in line] + [char_dict['<eos>']]

                lensen = len(sen2id)
                if lensen > max_len:
                    max_len = lensen

                length_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[lensen]))

                sen_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in
                               sen2id]

                seq_example = tf.train.SequenceExample(
                    context=tf.train.Features(feature={
                        'length': length_feature
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
