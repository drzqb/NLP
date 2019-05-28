import pycorrector
import pypinyin
from pypinyin import lazy_pinyin, Style
from xpinyin import Pinyin
import jieba

# print(jieba.lcut('王小明感帽了'))
# print(jieba.lcut('普通股股东的权益受到了侵害'))
# print(jieba.lcut('普通股东的权益受到了侵害'))
# print(pycorrector.correct('王小明感帽了'))
# print(pycorrector.correct('对京东新人度大打折扣'))
# print(pycorrector.ngram_score('普通股股东的权益受到了侵害'))
# print(pycorrector.ngram_score('普通股东的权益受到了侵害'))
# print(pycorrector.ngram_score('王小明感冒了'))
# print(pycorrector.get_same_pinyin('感冒'))
# print(pypinyin.pinyin('对京东新人度大打折扣', style=Style.NORMAL))
#
# p = Pinyin()
# print(p.get_pinyin('对京东新人度大打折扣'))

print(jieba.lcut('云计算是一项新技术'))

jieba.suggest_freq('云计算',tune=True)
print(jieba.lcut('云计算是一项新技术'))

# jieba.load_userdict('newdict.txt')
# print(jieba.lcut('云计算是一项新技术'))
