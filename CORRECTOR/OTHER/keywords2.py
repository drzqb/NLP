# coding=gbk
'''
    �ؼ�����ȡ
'''

import jieba
import jieba.analyse as kw

jb_funcs = []
sentence = 'ȫ���۰��о���᳤�����ڻ��Ϸ���ָ����ѧϰϵ����Ҫ����Ҫ��������ϯ������ۻع������ƻ�����������������������ǹ�ȥ20���ر����й�ʮ�˴�����''һ������''�����ʵ��ȡ�óɹ��ĸ������顣���ȣ�Ҫ�ں�ʵ��۵����ƻ�����������۵�����������������ֻ������ȷ����''һ������''���������򣬲��ܱ�֤''һ������''ʵ�������� �������Ρ���Σ�Ҫ�����ƻ�����ʵʩ���ƶȺͻ������ù�������ֱ����ʹ��Ȩ���������߶�����Ȩ�Ľ���������������򲻿ɻ�ȱ���������棬ͬʱ������ʵ��������������Ϊ���ĵ������������ơ�������Ҫ��ʵ��ǿ�������ر�����Թ�ְ��Ա����������ܷ����������������ι�����''һ��''��ʶ������''һ��''ԭ�򡣵��ģ�ҪŬ����ȫ����γɾ۽���չ���� �Ʒ����λ��ķ�Χ�����ܣ�ȫ��׼ȷ������ʵ�������йؾ�������Ĺ涨��ʹ��ۼ����ڹ��ҷ�չ�з��Ӷ������ò��� �����������ڻ�ø�ʵ�ڵ����档'


def run(func):
    jb_funcs.append(func)


def run_funcs():
    for func in jb_funcs:
        func()


@run
def tf_idf():
    # TF-IDF
    keywords = kw.extract_tags(sentence, topK=20, withWeight=True, allowPOS=('n', 'vn', 'ns', 'v'))
    print('TF-IDF�ؼ�����ȡ��{k}'.format(k=keywords))


@run
def textrank():
    # TextRank
    keywords = kw.textrank(sentence, topK=20, withWeight=True, allowPOS=('n', 'ns', 'vn', 'v'))
    print('TextRank�ؼ�����ȡ��{k}'.format(k=keywords))


if __name__ == '__main__':
    run_funcs()
