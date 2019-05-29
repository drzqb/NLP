'''
    结巴加载自定义字典同时调整词频后进行分词
'''
import jieba

jieba.load_userdict('dict.txt')

print(' '.join(jieba.lcut('台中正确应该不会被切开')))

fp = open('dict.txt', 'r', encoding='utf-8')
for line in fp:
    jieba.suggest_freq(line.strip(), tune=True)

print(' '.join(jieba.lcut('台中正确应该不会被切开')))
