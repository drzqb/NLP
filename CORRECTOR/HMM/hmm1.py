from hmmlearn import hmm
import numpy as np

hidden_state = ['box1', 'box2', 'box3']
n_state = len(hidden_state)
observation_state = ['red', 'white']
n_observation = len(observation_state)

pi = np.array([0.2, 0.4, 0.4])
A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])

model = hmm.MultinomialHMM(n_components=n_state)
model.startprob_ = pi
model.transmat_ = A
model.emissionprob_ = B
seen = np.array([[0, 1, 0]]).T

logprob, box = model.decode(seen, algorithm='viterbi')
print(logprob)
print("The ball picked:", ", ".join(map(lambda x: observation_state[x], np.reshape(seen, [-1]))))
print('The hidden Box;', ','.join(map(lambda x: hidden_state[x], np.reshape(box, [-1]))))

print('---------------------------------------------')

box2 = model.predict(seen)
print("The ball picked:", ", ".join(map(lambda x: observation_state[x], np.reshape(seen, [-1]))))
print('The hidden Box;', ','.join(map(lambda x: hidden_state[x], np.reshape(box2, [-1]))))

print('---------------------------------------------')

print(model.score(seen))


model2 = hmm.MultinomialHMM(n_components=n_state, n_iter=20, tol=0.01)
X2 = np.array([[0, 1, 0, 1], [0, 0, 0, 1], [1, 0, 1, 1]])
model2.fit(X2)
print(model2.startprob_)
print(model2.transmat_)
print( model2.emissionprob_)
print(model2.score(X2))
model2.fit(X2)
print(model2.startprob_)
print(model2.transmat_)
print( model2.emissionprob_)
print(model2.score(X2))
model2.fit(X2)
print(model2.startprob_)
print(model2.transmat_)
print( model2.emissionprob_)
print(model2.score(X2))
