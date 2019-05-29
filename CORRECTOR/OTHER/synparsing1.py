'''
    英文句法分析
'''
import spacy

nlp = spacy.load('en')

doc = nlp("You're best. it's word tokenization test for spacy. I love these books.")
print(doc)

for d in doc:
    print(d, d.lemma_, d.lemma)

for d in doc:
    print(d, d.pos_, d.pos)

for d in doc:
    print(d, d.tag_, d.tag)

# 依存分析
for d in doc:
    print(d, ':', str(list(d.children)))
