import tensorflow as tf
import collections
import pickle
import numpy as np
import sys

tf.flags.DEFINE_integer('most_common', 50000, 'number of most common used words')

CONFIG = tf.flags.FLAGS


class Lang():
    def __init__(self, config):
        self.config = config
        sentences, labels = self.read_data()
        character_dict, label_dict = self.make_dict(sentences, labels)
        print(len(character_dict))
        print(character_dict)
        print(len(character_dict))
        print(label_dict)
        self.toid(sentences, labels, character_dict, label_dict)

    def read_data(self):
        sentences = []
        labels = []

        DELIMETER = ' '

        with  open('corpus/corpus.txt', 'r', encoding='utf-8')as f:
            for line in f:
                line = line.strip().split(DELIMETER)
                sentence = [l.split('|')[0] for l in line]
                label = [l.split('|')[1] for l in line]

                sentences.append(sentence)
                labels.append(label)

        return sentences, labels

    def make_dict(self, sentences, labels):
        m_samples = len(sentences)

        tmp_characters = []
        tmp_labels = []
        for i in range(m_samples):
            tmp_characters.extend(sentences[i])
            tmp_labels.extend(labels[i])

        counter_characters = collections.Counter(tmp_characters).most_common(self.config.most_common - 2)
        character_dict = dict()
        character_dict['<pad>'] = len(character_dict)
        character_dict['<unknown>'] = len(character_dict)
        for character, _ in counter_characters:
            character_dict[character] = len(character_dict)

        counter_labels = collections.Counter(tmp_labels).most_common()
        label_dict = dict()
        label_dict['<pad>'] = len(label_dict)
        label_dict['<unknown>'] = len(label_dict)

        for label, _ in counter_labels:
            label_dict[label] = len(label_dict)

        with open('corpus/character_dict.txt', 'wb') as f:
            pickle.dump(character_dict, f)

        with open('corpus/label_dict.txt', 'wb') as f:
            pickle.dump(label_dict, f)

        return character_dict, label_dict

    def toid(self, sentences, labels, character_dict, label_dict):
        writer = tf.python_io.TFRecordWriter('corpus/train.tfrecord')
        m_samples = len(sentences)
        r = np.random.permutation(m_samples)
        sentences = [sentences[i] for i in r]
        labels = [labels[i] for i in r]

        max_len = 0
        for i in range(m_samples):
            sys.stdout.write('\r>> %d/%d' % (i + 1, m_samples))
            sys.stdout.flush()
            sen2id = [character_dict[character] if character in character_dict.keys() else character_dict['<unknown>']
                      for character in sentences[i]]
            label2id = [
                label_dict[labels[i][j]] if sen2id[j] != character_dict['<unknown>'] else label_dict['<unknown>']
                for j in range(len(labels[i]))]

            if len(sen2id) != len(label2id):
                print('error')
                exit()
            if len(sen2id) > max_len:
                max_len = len(sen2id)

            length_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[len(sen2id)]))

            sen_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in sen2id]
            label_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[label_])) for label_ in label2id]

            seq_example = tf.train.SequenceExample(
                context=tf.train.Features(feature={
                    'length': length_feature
                }),
                feature_lists=tf.train.FeatureLists(feature_list={
                    'sen': tf.train.FeatureList(feature=sen_feature),
                    'label': tf.train.FeatureList(feature=label_feature)
                })
            )

            serialized = seq_example.SerializeToString()

            writer.write(serialized)
        print('\n')
        print(max_len)


def main(unused_argv):
    lang = Lang(CONFIG)


if __name__ == '__main__':
    tf.app.run()
