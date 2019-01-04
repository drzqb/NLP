import tensorflow as tf
import collections
import re
import pickle

tf.flags.DEFINE_integer('most_common', 50000, 'number of most common used words')

CONFIG = tf.flags.FLAGS


class Lang():
    def __init__(self, config):
        self.config = config
        sentences, taggers = self.read_data()
        word_dict, tagger_dict = self.make_dict(sentences, taggers)
        self.toid(sentences, taggers, word_dict, tagger_dict)

    def read_data(self):
        sentences = []
        taggers = []
        with  open('data/199801.txt', encoding='gbk')as f:
            for line in f.readlines():
                if line.strip() != '':
                    data = re.sub('  ', ' ', line.strip())
                    data = re.sub('  ', ' ', data).split(' ')
                    word = []
                    tagger = []
                    for word_tag in data:
                        word_and_tag = word_tag.split('/')
                        word.append(word_and_tag[0])
                        tagger.append(word_and_tag[1])
                    sentences.append(word)
                    taggers.append(tagger)

        return sentences, taggers

    def make_dict(self, sentences, taggers):
        m_samples = len(sentences)

        tmp_words = []
        tmp_taggers = []
        for i in range(m_samples):
            tmp_words.extend(sentences[i])
            tmp_taggers.extend(taggers[i])

        counter_words = collections.Counter(tmp_words).most_common(self.config.most_common - 2)
        word_dict = dict()
        word_dict['<pad>'] = len(word_dict)
        word_dict['<unknown>'] = len(word_dict)
        for word, _ in counter_words:
            word_dict[word] = len(word_dict)

        counter_taggers = collections.Counter(tmp_taggers).most_common()
        tagger_dict = dict()
        tagger_dict['<pad>'] = len(tagger_dict)
        tagger_dict['<unknown>'] = len(tagger_dict)

        for tagger, _ in counter_taggers:
            tagger_dict[tagger] = len(tagger_dict)

        return word_dict, tagger_dict

    def toid(self, sentences, taggers, word_dict, tagger_dict):
        writer1 = tf.python_io.TFRecordWriter('data/train.tfrecord')
        writer2 = tf.python_io.TFRecordWriter('data/valid.tfrecord')
        writer3 = tf.python_io.TFRecordWriter('data/test.tfrecord')
        m_samples = len(sentences)
        number = [15000, 18000, m_samples]

        for i in range(m_samples):
            sen2id = [word_dict[word] if word in word_dict.keys() else word_dict['<unknown>'] for word in sentences[i]]
            tag2id = [tagger_dict[taggers[i][j]] if sen2id[j] != word_dict['<unknown>'] else tagger_dict['<unknown>']
                      for j in range(len(taggers[i]))]

            if len(sen2id)!=len(tag2id):
                print('error')
                exit()
            length_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[len(sen2id)]))

            sen_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in sen2id]
            tag_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[tag_])) for tag_ in tag2id]

            seq_example = tf.train.SequenceExample(
                context=tf.train.Features(feature={
                    'length': length_feature
                }),
                feature_lists=tf.train.FeatureLists(feature_list={
                    'sen': tf.train.FeatureList(feature=sen_feature),
                    'tag': tf.train.FeatureList(feature=tag_feature)
                })
            )

            serialized = seq_example.SerializeToString()
            if i < number[0]:
                writer1.write(serialized)
            elif i < number[1]:
                writer2.write(serialized)
            else:
                writer3.write(serialized)

        with open('data/word_dict.txt', 'wb') as f:
            pickle.dump(word_dict, f)

        with open('data/tagger_dict.txt', 'wb') as f:
            pickle.dump(tagger_dict, f)


def main(unused_argv):
    lang = Lang(CONFIG)


if __name__ == '__main__':
    tf.app.run()
