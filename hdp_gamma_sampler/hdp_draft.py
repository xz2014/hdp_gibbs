#! /usr/bin/env python
import sys

corpus=[np.array([0,0,1,1]),np.array([1]*4)]


V = 3
T = 2
alpha = 0.1
beta = [0.1]*V
m = 1
sizeofstack = 50
maxNIter = 100
numIter = 100
maxBufferSize = 5

D = len(corpus)
topics = range(T)

S_d = [np.random.choice(corpus[d],sizeofstack,replace=True).tolist() for d in range(D)] # stack for each document
B_dk = [[[] for k in topics] for a in corpus] # buffer
X_dk = [[[] for k in topics] for a in corpus] # each X_dk[d][k] is a list of words
n_dk =[np.array([0]*T) for a in corpus]

for d in range(D):
	for k in range(T):
		z = [x for x in corpus[d] if initial_topics[d][k] == k ]
		X_dk[d][k].extend(z)
		n_dk[d][k] = len(z)


for d in range(D):
	for k in range(k):
		for iter in range(1,maxNIter):
			newn =sample_ndk(d,k, maxBufferSize)
			print("The sampling path n_dk in document {} with topic {} is {}".format(d,k,newn))


