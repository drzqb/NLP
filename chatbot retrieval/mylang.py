import tensorflow as tf
import numpy as np
import csv
from nltk.tokenize import WordPunctTokenizer
import collections
import sys
import os
import pickle

tf.flags.DEFINE_integer('most_common', 100000, 'number of most common used words')

CONFIG = tf.flags.FLAGS


class Lang():
    def __init__(self, config):
        self.config = config

        self.train_process('data/train.csv')

        self.other_process('data/valid.csv')

        self.other_process('data/test.csv')

    def train_process(self, filename):
        print('reading train data...')
        train_data = []
        with open(filename) as f:
            csv_reader = csv.reader(f)
            title_reader = next(csv_reader)
            k = 1
            for row in csv_reader:
                if k <= 10000000:
                    train_data.append([WordPunctTokenizer().tokenize(row[0].lower().strip(' ')),
                                       WordPunctTokenizer().tokenize(row[1].lower().strip(' ')), row[2]])
                else:
                    break
                k += 1
        print('number of data :', k - 1)

        print('creating word_dict...')
        word_dict = {}

        tmp = []
        for sentences in train_data:
            tmp.extend(sentences[0])
            tmp.extend(sentences[1])

        word_dict['<unknown>'] = len(word_dict)

        counter = collections.Counter(tmp).most_common(self.config.most_common - 1)

        for word, _ in counter:
            word_dict[word] = len(word_dict)

        with open("data/word_dict.txt", "wb") as f:
            pickle.dump(word_dict, f)

        print('making tfrecord file...')
        writer = tf.python_io.TFRecordWriter('data/train.tfrecord')
        m_samples = len(train_data)
        for i in range(m_samples):
            tmp = train_data[i]
            context = [word_dict[word] if word in word_dict.keys() else word_dict['<unknown>'] for word in tmp[0]]
            utterance = [word_dict[word] if word in word_dict.keys() else word_dict['<unknown>'] for word in
                         tmp[1]]
            label = float(tmp[2])
            context_len = len(tmp[0])
            utterance_len = len(tmp[1])

            # 非序列化
            label_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
            length1_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[context_len]))
            length2_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[utterance_len]))
            # 序列化
            frame1_feature = [
                tf.train.Feature(int64_list=tf.train.Int64List(value=[frame_])) for frame_ in context
            ]
            frame2_feature = [
                tf.train.Feature(int64_list=tf.train.Int64List(value=[frame_])) for frame_ in utterance
            ]

            seq_example = tf.train.SequenceExample(
                # context 来放置非序列化部分
                context=tf.train.Features(feature={
                    "label": label_feature,
                    'context_len': length1_feature,
                    'utterance_len': length2_feature
                }),
                # feature_lists 放置变长序列
                feature_lists=tf.train.FeatureLists(feature_list={
                    "context": tf.train.FeatureList(feature=frame1_feature),
                    'utterance': tf.train.FeatureList(feature=frame2_feature)
                })
            )

            serialized = seq_example.SerializeToString()
            writer.write(serialized)

    def other_process(self, filename):
        with open('data/word_dict.txt', 'rb') as f:
            word_dict = pickle.load(f)

        print('saving ' + filename.split('/')[-1].split('.')[0] + ' data as tfrecord')
        writer = tf.python_io.TFRecordWriter(filename.split('.')[0] + '.tfrecord')

        with open(filename) as f:
            csv_reader = csv.reader(f)
            title_reader = next(csv_reader)
            k = 1

            for row in csv_reader:
                if k <= 1000:
                    tmp = [WordPunctTokenizer().tokenize(row[i].lower().strip(' ')) for i in range(len(row))]
                    for j in range(1, 11):
                        context = [word_dict[word] if word in word_dict.keys() else word_dict['<unknown>']
                                   for word in tmp[0]]
                        utterance = [word_dict[word] if word in word_dict.keys() else word_dict['<unknown>']
                                     for word in tmp[j]]

                        if j == 1:
                            label = 1.0
                        else:
                            label = 0.0
                        context_len = len(tmp[0])
                        utterance_len = len(tmp[j])

                        # 非序列化
                        label_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
                        length1_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[context_len]))
                        length2_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[utterance_len]))
                        # 序列化
                        frame1_feature = [
                            tf.train.Feature(int64_list=tf.train.Int64List(value=[frame_])) for frame_ in context
                        ]
                        frame2_feature = [
                            tf.train.Feature(int64_list=tf.train.Int64List(value=[frame_])) for frame_ in utterance
                        ]

                        seq_example = tf.train.SequenceExample(
                            # context 来放置非序列化部分
                            context=tf.train.Features(feature={
                                "label": label_feature,
                                'context_len': length1_feature,
                                'utterance_len': length2_feature
                            }),
                            # feature_lists 放置变长序列
                            feature_lists=tf.train.FeatureLists(feature_list={
                                "context": tf.train.FeatureList(feature=frame1_feature),
                                'utterance': tf.train.FeatureList(feature=frame2_feature)
                            })
                        )

                        serialized = seq_example.SerializeToString()
                        writer.write(serialized)
                else:
                    break
                k += 1
        print('number of data :', k - 1)



def main(unused_argv):
    lang = Lang(CONFIG)


if __name__ == '__main__':
    tf.app.run()
