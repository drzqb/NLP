'''
    nltk 英文分词
'''
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet

text = 'I am a student. They are doctors. You are doing your homework'
nltk_func = []


def run(func):
    nltk_func.append(func)


def run_funcs():
    for func in nltk_func:
        func()


@run
def func1():
    value = nltk.sent_tokenize(text)
    print(value)

    words = [nltk.word_tokenize(sen) for sen in value]
    print(words)


@run
def func2():
    words = nltk.word_tokenize(text)
    print(words)
    tags = nltk.pos_tag(words)
    print(tags)


@run
def func3():
    words = nltk.word_tokenize(text)
    ps = PorterStemmer()
    origin_word = [ps.stem(word) for word in words]
    print('原始形式：{ow}'.format(ow=origin_word))


@run
def func4():
    words = wordnet.synsets('happy')
    for w in words:
        for l in w.lemmas():
            print('同义词：{w}'.format(w=l.name()))
            if l.antonyms():
                print('反义词：{w}'.format(w=l.antonyms()[0].name()))


if __name__ == '__main__':
    run_funcs()
