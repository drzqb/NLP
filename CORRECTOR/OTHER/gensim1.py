'''
    使用gensim训练Word2Vec
'''

from gensim.models import word2vec
import jieba
import jieba.analyse

jieba.suggest_freq('沙瑞金', True)
jieba.suggest_freq('田国富', True)
jieba.suggest_freq('高育良', True)
jieba.suggest_freq('侯亮平', True)
jieba.suggest_freq('钟小艾', True)
jieba.suggest_freq('陈岩石', True)
jieba.suggest_freq('欧阳菁', True)
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('蔡成功', True)
jieba.suggest_freq('孙连城', True)
jieba.suggest_freq('季昌明', True)
jieba.suggest_freq('丁义珍', True)
jieba.suggest_freq('郑西坡', True)
jieba.suggest_freq('赵东来', True)
jieba.suggest_freq('高小琴', True)
jieba.suggest_freq('赵瑞龙', True)
jieba.suggest_freq('林华华', True)
jieba.suggest_freq('陆亦可', True)
jieba.suggest_freq('刘新建', True)
jieba.suggest_freq('刘庆祝', True)

with open('data/in_the_name_of_people.txt', encoding='utf-8') as f:
    document = f.read()
    document_cut = jieba.cut(document)
    result = ' '.join(document_cut)
    with open('data/in_the_name_of_people_segment.txt', 'w', encoding='utf-8') as f2:
        f2.write(result)

sentences = word2vec.LineSentence('data/in_the_name_of_people_segment.txt')
model = word2vec.Word2Vec(sentences, window=3, hs=1, min_count=1, size=100)

print(model.similarity('早饭', '晚饭'))

words = model.most_similar('几乎')
for word in words:
    print(word[0], word[1])
