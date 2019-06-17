import re
import jieba
import numpy as np
from xpinyin import Pinyin
import Pinyin2Hanzi
from Pinyin2Hanzi import dag
from tqdm import tqdm
from string import punctuation as en
from zhon.hanzi import punctuation as ch

# P： 拼音相似 80%    S：字形相似 10%    R：多字 5%   L：词缺字 4%  O：正确 80%
ERROR_MODE = {'pinyin': 'P',
              'reduction': 'R',
              'loss': 'L',
              'shape': 'S',
              'right': 'O'}

MAX_ERROR_NUM = 20
DELIMETER = ' '
Max_PATH_NUM = 20
COMMON_DICT = ['的', '地', '得', '了', '个', '子', '只']


def read_data(filename):
    raw_ch = []
    s = open(filename, 'r', encoding='utf-8').read()
    s = s.split('\n')

    for sl in s[:]:
        tmp_sl = sl.strip(' ')
        tmp_sl = re.sub(' ', '', tmp_sl)
        tmp_sl = re.sub('\t', '', tmp_sl)
        if tmp_sl == '':
            s.remove(sl)

    for sl in s[0:-1:2]:
        raw_ch.append(re.sub(' |\u3000', '', sl.strip(' ')))

    return raw_ch


def write_data(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for line in data:
            sentences = re.sub('[{}]'.format(en + ch), DELIMETER, line)
            sentences = re.split(DELIMETER, sentences)  # 保留分割符
            for sent in sentences:
                if len(jieba.lcut(sent)) > 2:
                    f.write(sent + '\n')


def build_corpus(fromfile, tofile):
    fc = open(tofile, 'w', encoding='utf-8')
    p = Pinyin()
    dagParams = Pinyin2Hanzi.DefaultDagParams()

    print('Begin to build corpus...')
    with open(fromfile, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            # 正确语句
            line = line.strip()
            kline = len(line)
            for l in range(kline):
                fc.write(line[l] + '|' + ERROR_MODE['right'])
                if l < kline - 1:
                    fc.write(DELIMETER)
            fc.write('\n')

            # 构造一个错误
            ERROR_NUM = np.random.randint(10, MAX_ERROR_NUM) + 1
            for n in range(ERROR_NUM):
                #  P： 拼音相似 80%    S：字形相似 10%    R：多字 5%   L：词缺字 5%    O：正确 80%
                words = jieba.lcut(line)
                kwords = len(words)
                k1 = np.random.randint(kwords)  # 随机选某一个位置的词
                k2 = np.random.randint(len(ERROR_MODE) - 2)

                if k2 == 0:  # 拼音相似
                    word = words[k1]
                    result = p.get_pinyin(word)
                    PATH_NUM = np.random.randint(10, Max_PATH_NUM) + 1
                    similars = dag(dagParams, result.split('-'), path_num=PATH_NUM)

                    if len(similars) > 0:
                        for pn in range(len(similars)):
                            if ''.join(similars[pn].path) != word:
                                words[k1] = ''.join(similars[pn].path)

                                for l in range(kwords):
                                    for ll in range(len(words[l])):
                                        if l != k1:
                                            fc.write(words[l][ll] + '|' + ERROR_MODE['right'])
                                        else:
                                            fc.write(words[l][ll] + '|' + ERROR_MODE['pinyin'])
                                        if l < kwords - 1:
                                            fc.write(DELIMETER)
                                        else:
                                            if ll < len(words[l]) - 1:
                                                fc.write(DELIMETER)
                                fc.write('\n')

                elif k2 == 1:  # 某词后面多字
                    for com in COMMON_DICT:
                        if k1 < kwords - 1:
                            for l in range(kwords):
                                if l <= k1:
                                    for ll in range(len(words[l])):
                                        fc.write(words[l][ll] + '|' + ERROR_MODE['right'] + DELIMETER)
                                else:
                                    for ll in range(len(words[l])):
                                        if (l < kwords - 1) or (l == kwords - 1 and ll < len(words[l]) - 1):
                                            fc.write(words[l][ll] + '|' + ERROR_MODE['right'] + DELIMETER)
                                        else:
                                            fc.write(words[l][ll] + '|' + ERROR_MODE['right'])
                                if l == k1:
                                    fc.write(com + '|' + ERROR_MODE['reduction'] + DELIMETER)
                        else:
                            for l in range(kwords):
                                for ll in range(len(words[l])):
                                    fc.write(words[l][ll] + '|' + ERROR_MODE['right'] + DELIMETER)
                            fc.write(com + '|' + ERROR_MODE['reduction'])
                        fc.write('\n')

                elif k2 == 2:  # 词缺字（最后一个字）
                    if len(words[k1]) > 1:
                        words[k1] = words[k1][:-1]
                        for l in range(kwords):
                            for ll in range(len(words[l])):
                                if l != k1:
                                    fc.write(words[l][ll] + '|' + ERROR_MODE['right'])
                                else:
                                    fc.write(words[l][ll] + '|' + ERROR_MODE['loss'])
                                if l < kwords - 1:
                                    fc.write(DELIMETER)
                                else:
                                    if ll < len(words[l]) - 1:
                                        fc.write(DELIMETER)
                        fc.write('\n')

            # 构造两个错误
            if len(jieba.lcut(line)) >= 5:
                ERROR_NUM = np.random.randint(10, MAX_ERROR_NUM) + 1
                for n in range(ERROR_NUM):
                    #  P： 拼音相似 80%    S：字形相似 10%    R：多字 5%   L：词缺字 5%    O：正确 80%
                    words = jieba.lcut(line)
                    kwords = len(words)
                    k1 = np.sort(np.random.permutation(kwords)[:2])  # 随机选某两个位置的词
                    k2 = np.random.randint(len(ERROR_MODE) - 2, size=[2])  # 随机选取两种错误类型

                    errwords = ''
                    SUCCEED = True
                    if (k2[0] == 2 and len(words[k1[0]]) < 2) or (k2[1] == 2 and len(words[k1[1]]) < 2):
                        SUCCEED = False
                    else:
                        for l in range(kwords):
                            word = words[l]
                            if l != k1[0] and l != k1[1]:
                                for ll in range(len(word)):
                                    errwords += word[ll] + '|' + ERROR_MODE['right']
                                    if l < kwords - 1:
                                        errwords += DELIMETER
                                    else:
                                        if ll < len(word) - 1:
                                            errwords += DELIMETER
                            else:
                                if l == k1[0]:
                                    k = 0
                                else:
                                    k = 1

                                if k2[k] == 0:  # 拼音相似
                                    result = p.get_pinyin(word)
                                    PATH_NUM = np.random.randint(10, Max_PATH_NUM) + 1
                                    similars = dag(dagParams, result.split('-'), path_num=PATH_NUM)

                                    if len(similars) == 0:
                                        SUCCEED = False
                                    else:
                                        ks = np.random.randint(len(similars))
                                        if ''.join(similars[ks].path) == word:
                                            SUCCEED = False
                                        else:
                                            word = ''.join(similars[ks].path)

                                            for ll in range(len(word)):
                                                errwords += word[ll] + '|' + ERROR_MODE['pinyin']
                                                if l < kwords - 1:
                                                    errwords += DELIMETER
                                                else:
                                                    if ll < len(word) - 1:
                                                        errwords += DELIMETER
                                elif k2[k] == 1:  # 某词后面多字
                                    for ll in range(len(word)):
                                        errwords += word[ll] + '|' + ERROR_MODE['right'] + DELIMETER

                                    errwords += np.random.choice(COMMON_DICT) + '|' + ERROR_MODE['reduction']
                                    if l < kwords - 1:
                                        errwords += DELIMETER
                                elif k2[k] == 2:  # 词缺字（最后一个字）
                                    for ll in range(len(word) - 1):
                                        errwords += word[ll] + '|' + ERROR_MODE['loss']
                                        if l < kwords - 1:
                                            errwords += DELIMETER
                                        else:
                                            if ll < len(word) - 2:
                                                errwords += DELIMETER

                    if SUCCEED:
                        fc.write(errwords + '\n')


if __name__ == '__main__':
    text = read_data('corpus/bps.txt')
    write_data(text, 'corpus/bps1.txt')
    build_corpus('corpus/bps1.txt', 'corpus/corpus.txt')
