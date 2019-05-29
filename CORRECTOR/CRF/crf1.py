'''
    CRF for NER
'''
import matplotlib.pyplot as plt

plt.style.use('ggplot')
from itertools import chain
import nltk
import scipy.stats
import sklearn
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

import sklearn_crfsuite as CRF
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

# nltk.download('conll2002')

print(nltk.corpus.conll2002.fileids())

train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
print(train_sents[0])

def word2features(sent,i):
    word=sent[i][0]
    postag=sent[i][1]

    features=
# model=CRF.CRF()
# model.fit()
