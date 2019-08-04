import tensorflow as tf
import re
from tqdm import tqdm
from chardict import load_vocab
import xlrd

tf.flags.DEFINE_integer('maxword', 512, 'max length of any sentences')

CONFIG = tf.flags.FLAGS


def replace(txt):
    txt = re.sub('“', '"', txt.lower())
    txt = re.sub('”', '"', txt)
    return txt


class Lang():
    def __init__(self, config):
        self.config = config
        # self.read_data()
        self.toid()

    def read_data(self):
        fc = open('data/finance1.txt', 'w', encoding='utf-8')
        m_samples = 0
        workbook = xlrd.open_workbook("data/finance.xlsx")

        worksheet = workbook.sheet_by_index(0)
        nrows = worksheet.nrows  # 获取该表总行数
        ncols = worksheet.ncols  # 获取该表总列数

        for i in range(nrows - 1):
            # sentences = re.split('([。])', worksheet.cell_value(i + 1, 2).strip())
            # sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2])]
            para = re.sub('([。！？\?])([^”’])', r"\1\n\2", worksheet.cell_value(i + 1, 2).strip())  # 单字符断句符
            para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
            para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
            para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
            # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
            para = para.rstrip()  # 段尾如果有多余的\n就去掉它
            # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
            sentences = para.split("\n")
            for sentence in sentences:
                sentence = sentence.strip()
                # if not sentence.endswith( '。'):
                #     sentence += '。'
                if len(sentence) <= self.config.maxword - 1 and len(sentence) > 1:
                    fc.write(replace(sentence) + '\n')
                    m_samples += 1

        fc.close()
        print('样本总量共：%d 句' % m_samples)  # 488547 句

    def toid(self):
        char_dict = load_vocab('vocab.txt')

        writer = tf.python_io.TFRecordWriter('data/train_finance1.tfrecord')
        max_len = 0
        with  open('data/finance1.txt', 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip()

                sen2id = [char_dict['[CLS]']] + [
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
