import argparse
import jieba
import collections
import numpy as np
import pickle

parser = argparse.ArgumentParser(description='manual to this script')

parser.add_argument('--corpus', type=str, default='data/xiaohuangji50w_nofenci.conv',
                    help='The corpus file path')
parser.add_argument('--most_qa', type=int, default=10000,
                    help='The max length of QA dictionary')

params = parser.parse_args()


class Lang():
    '''
    读取聊天语料并建立字典
    '''

    def __init__(self):
        self.read_data()

        self.create_dict()
        self.text2id()

    # 读取聊天语料
    def read_data(self):
        self.raw_q = []
        self.raw_a = []
        s = open(params.corpus, 'r', encoding='utf-8').read()
        s = s.split('\n')

        print('jieba...')
        k = 1
        for sl in s[:]:
            if k == 20001:
                break
            if sl.startswith('M '):
                sl = sl.strip('M ')
                if sl == '':
                    sl = '？？？'
                if k % 2 == 1:
                    self.raw_q.append(jieba.lcut(sl))
                else:
                    self.raw_a.append(jieba.lcut(sl))

                k += 1

        print('removing long sentences......')
        tmp_q = self.raw_q.copy()
        tmp_a = self.raw_a.copy()
        for i in reversed(range(len(tmp_q))):
            if len(tmp_q[i]) > 20 or len(tmp_a[i]) >= 20:
                self.raw_q.pop(i)
                self.raw_a.pop(i)

    # 建立字典
    def create_dict(self):
        self.qa_dict = dict()

        tmp_raw_qa = []
        for q in self.raw_q:
            tmp_raw_qa.extend(q)
        for a in self.raw_a:
            tmp_raw_qa.extend(a)
        counter = collections.Counter(tmp_raw_qa).most_common(params.most_qa - 4)

        self.qa_dict['<PAD>'] = len(self.qa_dict)
        self.qa_dict['<EOS>'] = len(self.qa_dict)
        self.qa_dict['<UNK>'] = len(self.qa_dict)
        self.qa_dict['<GO>'] = len(self.qa_dict)
        for word, _ in counter:
            self.qa_dict[word] = len(self.qa_dict)

        self.qa_reverse_dict = {v: k for k, v in self.qa_dict.items()}

    # 语料向量化
    def text2id(self):
        self.qtext2id = []
        self.qtext2id_length = []
        self.atext2id_target = []
        self.atext2id_input = []
        self.atext2id_input_length = []

        for q in self.raw_q:
            tmp = []
            for word in q:
                tmp.append(self.qa_dict[word] if word in self.qa_dict.keys() else self.qa_dict['<UNK>'])
            self.qtext2id.append(tmp)
            self.qtext2id_length.append(len(tmp))

        for a in self.raw_a:
            tmp = []
            for word in a:
                tmp.append(self.qa_dict[word] if word in self.qa_dict.keys() else self.qa_dict['<UNK>'])

            tmp1 = tmp.copy()
            tmp1.insert(0, self.qa_dict['<GO>'])
            self.atext2id_input.append(tmp1)
            self.atext2id_input_length.append(len(tmp1))

            tmp.append(self.qa_dict['<EOS>'])
            self.atext2id_target.append(tmp)


if __name__ == '__main__':
    lang = Lang()
    print(len(lang.qa_dict))
    print(len(lang.qtext2id), np.max(lang.qtext2id_length), np.min(lang.qtext2id_length))
    print(len(lang.atext2id_input), np.max(lang.atext2id_input_length), np.min(lang.atext2id_input_length))

    with open('data/qa_dict.txt', 'wb') as f:
        pickle.dump(lang.qa_dict, f)
    with open('data/qa_reverse_dict.txt', 'wb') as f:
        pickle.dump(lang.qa_reverse_dict, f)

    with open('data/qtext2id.txt', 'wb') as f:
        pickle.dump(lang.qtext2id, f)

    with open('data/atext2id_input.txt', 'wb') as f:
        pickle.dump(lang.atext2id_input, f)

    with open('data/atext2id_target.txt', 'wb') as f:
        pickle.dump(lang.atext2id_target, f)

    with open('data/xhj.txt','w+', encoding='utf-8') as f:
        for i in range(len(lang.qtext2id)):
            f.write(' '.join(lang.raw_q[i]) + '\n')
            f.write(' '.join(lang.raw_a[i]) + '\n')
            f.write('\n')
