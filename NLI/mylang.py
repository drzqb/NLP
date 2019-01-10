import json
import tensorflow as tf
from nltk.tokenize import WordPunctTokenizer
import pickle
import numpy as np

GLOVE_SIZE=[400000,50]

class Lang():
    def __init__(self):
        self.label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        with open('data/label_dict.txt', 'wb') as f:
            pickle.dump(self.label_dict, f)

        print('\ncreating word dict using pretrained glove vector...')
        self.create_dict()

        print('\npreprocessing train data...')
        phl = self.read_json('data/snli_1.0_train.jsonl', 500000)
        self.sen2id(phl, 'train')

        print('\npreprocessing valid data...')
        phl = self.read_json('data/snli_1.0_val.jsonl', 1000)
        self.sen2id(phl, 'val')

        print('\npreprocessing test data...')
        phl = self.read_json('data/snli_1.0_test.jsonl', 1000)
        self.sen2id(phl, 'test')

    def sen2id(self, phl, data_usage):
        p, h, l = phl

        tfrecord_name = 'data/' + data_usage + '.tfrecord'
        writer = tf.python_io.TFRecordWriter(tfrecord_name)

        m_samples = len(p)
        for i in range(m_samples):
            l_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[self.label_dict[l[i]]]))
            p_len_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[len(p[i])]))
            h_len_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[len(h[i])]))

            p_feature = [
                tf.train.Feature(int64_list=tf.train.Int64List(value=[self.word_dict.get(p_, self.word_dict['<unk>'])]))
                for p_ in p[i]]
            h_feature = [
                tf.train.Feature(int64_list=tf.train.Int64List(value=[self.word_dict.get(h_, self.word_dict['<unk>'])]))
                for h_ in h[i]]

            seq_example = tf.train.SequenceExample(
                context=tf.train.Features(feature={
                    'label': l_feature,
                    'p_len': p_len_feature,
                    'h_len': h_len_feature
                }),
                feature_lists=tf.train.FeatureLists(feature_list={
                    'premise': tf.train.FeatureList(feature=p_feature),
                    'hypothesis': tf.train.FeatureList(feature=h_feature)
                })
            )
            serialized = seq_example.SerializeToString()
            writer.write(serialized)

    def create_dict(self):
        embed_matrix = np.zeros(GLOVE_SIZE, np.float32)

        self.word_dict = dict()
        self.word_dict['<pad>'] = 0
        self.word_dict['<unk>'] = 1
        with open('data/glove.6B.50d.txt', 'r') as f:
            for line in f.readlines():
                tmp = line.split(' ')
                embed_matrix[len(self.word_dict) - 2] = np.array([float(t) for t in tmp[1:]])
                self.word_dict[tmp[0]] = len(self.word_dict)

        with open('data/word_dict.txt', 'wb') as f:
            pickle.dump(self.word_dict, f)

        with open('data/embed_matrix.txt', 'wb') as f:
            pickle.dump(embed_matrix, f)

    def read_json(self, filename, kmax):
        p = []
        h = []
        l = []
        k = 0
        with open(filename, 'r') as f:
            for line in f.readlines():
                load_dict = json.loads(line)
                tmp = load_dict['gold_label']
                if tmp != '-' and k <= kmax:
                    p.append(load_dict['sentence1'])
                    h.append(load_dict['sentence2'])
                    l.append(tmp)
                    k += 1
        return p, h, l


def main(unused_argv):
    lang = Lang()


if __name__ == '__main__':
    tf.app.run()
