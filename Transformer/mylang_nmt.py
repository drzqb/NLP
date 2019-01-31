import tensorflow as tf
from nltk.tokenize import WordPunctTokenizer
import jieba
import collections
import re
import numpy as np
import pickle

tf.flags.DEFINE_string('corpus', 'data/bps.txt', 'The corpus file path')
tf.flags.DEFINE_integer('most_en', 5000, 'The max length of englishword dictionary')
tf.flags.DEFINE_integer('most_ch', 5000, 'The max length of chineseword dictionary')
FLAGS = tf.flags.FLAGS


class Lang():
    '''
    读取中英文语料并建立中英文字典
    '''

    def __init__(self, config):
        self.config = config
        self.read_data()

        self.create_dict()
        self.text2id()

    # 读取中英文语料
    def read_data(self):
        self.raw_en = []
        self.raw_ch = []
        s = open(self.config.corpus, 'r', encoding='utf-8').read()
        s = s.split('\n')

        for sl in s[:]:
            tmp_sl = sl.strip(' ')
            tmp_sl = re.sub(' ', '', tmp_sl)
            tmp_sl = re.sub('\t', '', tmp_sl)
            if tmp_sl == '':
                s.remove(sl)

        for sl in s[1:len(s):2]:
            sl = sl.lower().strip(' ')
            if not sl.endswith('.'):
                sl += '.'
            self.raw_en.append(WordPunctTokenizer().tokenize(sl))

        for sl in s[0:-1:2]:
            sl = sl.strip(' ')
            if not sl.endswith('。'):
                sl += '。'
            self.raw_ch.append(jieba.lcut(re.sub(' |\u3000', '', sl)))

        tmp_en = self.raw_en.copy()
        tmp_ch = self.raw_ch.copy()
        for i in reversed(range(len(tmp_en))):
            if len(tmp_en[i]) > 20:
                self.raw_en.pop(i)
                self.raw_ch.pop(i)

    # 建立中英文字典
    def create_dict(self):
        self.en_dict = dict()
        self.ch_dict = dict()

        tmp_raw_en = []
        for en in self.raw_en:
            tmp_raw_en.extend(en)
        counter = collections.Counter(tmp_raw_en).most_common(self.config.most_en - 2)

        self.en_dict['<PAD>'] = len(self.en_dict)
        self.en_dict['<UNK>'] = len(self.en_dict)
        for word, _ in counter:
            self.en_dict[word] = len(self.en_dict)

        self.en_reverse_dict = {v: k for k, v in self.en_dict.items()}

        with open('data/en_dict.txt', 'wb') as f:
            pickle.dump(self.en_dict, f)

        with open('data/en_reverse_dict.txt', 'wb') as f:
            pickle.dump(self.en_reverse_dict, f)

        tmp_raw_ch = []
        for ch in self.raw_ch:
            tmp_raw_ch.extend(ch)
        counter = collections.Counter(tmp_raw_ch).most_common(self.config.most_ch - 4)

        self.ch_dict['<PAD>'] = len(self.ch_dict)
        self.ch_dict['<EOS>'] = len(self.ch_dict)
        self.ch_dict['<UNK>'] = len(self.ch_dict)
        self.ch_dict['<GO>'] = len(self.ch_dict)
        for word, _ in counter:
            self.ch_dict[word] = len(self.ch_dict)

        self.ch_reverse_dict = {v: k for k, v in self.ch_dict.items()}

        with open('data/ch_dict.txt', 'wb') as f:
            pickle.dump(self.ch_dict, f)

        with open('data/ch_reverse_dict.txt', 'wb') as f:
            pickle.dump(self.ch_reverse_dict, f)

    # 语料向量化
    def text2id(self):
        self.entext2id = []
        self.entext2id_length = []

        self.chtext2id_target = []
        self.chtext2id_input = []
        self.chtext2id_input_length = []

        for en in self.raw_en:
            tmp = []
            for word in en:
                tmp.append(self.en_dict[word] if word in self.en_dict.keys() else self.en_dict['<UNK>'])
            self.entext2id.append(tmp)
            self.entext2id_length.append(len(tmp))

        for ch in self.raw_ch:
            tmp = []
            for word in ch:
                tmp.append(self.ch_dict[word] if word in self.ch_dict.keys() else self.ch_dict['<UNK>'])

            tmp1 = tmp.copy()
            tmp1.insert(0, self.ch_dict['<GO>'])
            self.chtext2id_input.append(tmp1)
            self.chtext2id_input_length.append(len(tmp1))

            tmp.append(self.ch_dict['<EOS>'])
            self.chtext2id_target.append(tmp)

        with open('data/entext2id.txt', 'wb') as f:
            pickle.dump(self.entext2id, f)

        with open('data/chtext2id_input.txt', 'wb') as f:
            pickle.dump(self.chtext2id_input, f)

        with open('data/chtext2id_target.txt', 'wb') as f:
            pickle.dump(self.chtext2id_target, f)


def main(unused_argv):
    lang = Lang(FLAGS)
    print(len(lang.raw_en), lang.raw_en[:10])
    print(len(lang.raw_ch), lang.raw_ch[:10])

    print(len(lang.en_dict), [lang.en_reverse_dict[i] for i in range(100)])
    print(len(lang.ch_dict), [lang.ch_reverse_dict[i] for i in range(100)])

    print(lang.entext2id_length[:10], np.max(lang.entext2id_length), np.min(lang.entext2id_length),
          np.argmin(lang.entext2id_length), lang.entext2id[:10])
    print(lang.chtext2id_input_length[:10], np.max(lang.chtext2id_input_length), np.min(lang.chtext2id_input_length),
          np.argmin(lang.chtext2id_input_length), lang.chtext2id_target[:10])
    print(lang.chtext2id_input_length[:10], np.max(lang.chtext2id_input_length), np.min(lang.chtext2id_input_length),
          np.argmin(lang.chtext2id_input_length), lang.chtext2id_input[:10])

    for i in range(len(lang.entext2id)):
        print(''.join(lang.raw_ch[i]))
        print(' '.join(lang.raw_en[i]))
    f = open('data/bps2.txt', 'w+')
    for i in range(len(lang.entext2id)):
        f.write(' '.join(lang.raw_ch[i]) + '\n')
        f.write(' '.join(lang.raw_en[i]) + '\n')
        f.write('\n')
    f.close()


if __name__ == '__main__':
    tf.app.run()
