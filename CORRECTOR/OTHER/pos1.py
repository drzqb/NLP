'''
    利用词性清洗部分词语，进行信息提取
'''
import jieba
import jieba.posseg as pseg

jieba.load_userdict('dict.txt')
filter=set(['p','x','uj','d','a'])
pseg1=pseg.lcut('恶性肿瘤的分期越高，患者预后越差。通过对肿瘤不同恶性程度的划分，TNM分期在预测预后方面更加有效。')
print(pseg1)
pos_filter=[ps.word for ps in pseg1 if ps.flag not in filter]

print(pos_filter)