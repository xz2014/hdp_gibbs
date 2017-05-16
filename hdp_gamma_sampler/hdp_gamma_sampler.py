#! /usr/bin/env python

#### #######################################  Gibbs sampler in HDP & Gamma-Gamma-Poisson-Process
import numpy as np
import scipy.special as sp
import scipy.misc as sm
#from __future__ import division


class hdp_GGPP:

	def __init__(self, alpha, beta, m, V, T, corpus,sizeofstack, maxNIter, maxBufferSize,numIter):
		self.alpha = alpha 
		self.beta = beta   
		self.V = V   
		self.m = m   # n_dk is poisson ~ ( m * pi_dk ) , empirically set as 1/T. 
		self.T = T   # number of topics
		self.corpus = corpus
		self.sizeofstack = sizeofstack # size of each stack for a document. 
		self.numIter = numIter
		self.maxNIter = maxNIter # maxNIter: number of iterations to update n_dk in each iteration
		self.maxBufferSize = maxBufferSize   # if length of B_dk exceeds maxBufferSize, push all the elements from B_dk to S_d

		self.D = len(corpus)
		self.topics = range(T)  # topics are indexed as 0,1,2...
		self.initial_topics = [[0]*len(corpus[0])]*2  # initial topic assignment

##### All variables:
		self.S_d = [np.random.choice(corpus[d],sizeofstack,replace=True).tolist() for d in range(self.D)] # the stack for each document
		self.B_dk = [[[] for k in self.topics] for a in corpus] # buffer
		self.X_dk = [[[] for k in self.topics] for a in corpus] # each X_dk[d][k] is a list of words
		self.n_dk =[[0]*T for a in corpus] # each n_dk[d][k] is the number of words in document d with topic k
		self.alpha_k = []*T
		self.theta_k = [[0.0]*V for a in self.topics]
		self.pi_dk = [[np.ndarray(0,dtype=int) for k in self.topics] for a in corpus]

#### Initialize each variable

	def initial_(self):
		for d in range(self.D):
			for k in range(self.T):
				z = [x for x in self.corpus[d] if self.initial_topics[d][k] == k ]
				self.X_dk[d][k].extend(z)
				self.n_dk[d][k] = len(z)
		self.pi_dk = [[1]*self.T for a in self.corpus]
		self.alpha_k = [1]*self.T
		self.theta_k = [[0.5]*self.V for a in self.topics]
		m = 1/self.T



######### for a given k, run gibbs sampling 

	def parallel_samplging_for_each_topic(self,k): 
		self.initial_()
		new_n_with_topic_k=[]
		L=[]

		for t in range(1,self.numIter):
			for d in range(self.D):
				# update n_dk for each d and k
				n_dk[d][k] = self.updating_ndk(d, k)
				
				# update pi_dk for each d and k
				pi_dk[d][k] = np.random.gamma(n_dk[d][k]+alpha_k[k], m+1)

			# update alpha_k for each k
			alpha_k = np.random.gamma(self.alpha/self.T+self.D,-sum(np.log(zip(*pi_dk)[k])))

			# update theta_k for each k
			prob_theta_k = np.exp(sum((self.beta+self.n_kw[k])*np.log(theta_k[k])))
			theta_k = np.random.dirichlet(np.add(self.beta,self.n_kw[k]))

			# compute log likelihood
			l = self.log_likelihood()
			L.append(l)
			print "The Log-likelihood at iteration {}. is {}. Topic{}'.format(t,l,k)"

		return(L)


	

#### for a given d and k, update n_dk using metroplis hasting algorithm. maxNIter iterations

	def updating_ndk(self, d, k):
		for iter in range(1,self.maxNIter):
			newn =self.sample_ndk(d, k)
		return(newn)


####### sample n_dk, one iteration
	def sample_ndk(self, d,k):
		u = np.random.uniform(0,1)

		if u<0.5:
			x=np.random.choice(S_d[d],replace=False)
			S_d[d].remove(x)

			uu = np.random.uniform(0,1)

			if (uu < p_plus(x)): # accept with prob = min(1,p_plus)
				n_dk[d][k] += 1
				X_dk[d][k].append(x)
			else:
				B_dk[d][k].append(x)

		else:
			if (len(X_dk[d][k]) > 0):
				x=np.random.choice(X_dk[d][k],replace=False)
				X_dk[d][k].remove(x)
				uu = np.random.uniform(0,1)
				if (uu < p_minus(x)):  
					n_dk[d][k] -= 1
					B_dk[d][k].append(x)
				else:
					X_dk[d][k].append(x)

		if (len(B_dk[d][k]) > maxBufferSize):
			S_d[d].extend(B_dk[d][k])

		return(n_dk[d][k])




# acceptance rate 
	def p_minus(self,w):
		return(n_dk[d][k]/((m*pi_dk[d][k])*theta_k[k][w]))
		

	def p_plus(self,w):
		return((m*pi_dk[d][k])*theta_k[k][w]/(n_dk[d][k]+1))




# complete loglikelihood log P(n_dk, pi_dk, alpha_k, theta_k, X_dk)
	def log_likelihood(self):
		LL=[]
		for k in range(T):
			v = np.array(zip(*n_dk)[k])
			q = np.array(zip(*pi_dk)[k])
			e = np.array(zip(*X_dk)[k])
			l1 = sum(np.log(m)*v-sp.gammaln(alpha_k[k])+(v+alpha_k[k]-1)*np.log(q)-(m+1)*q-np.log(sm.factorial(v)))
			l2 = (-1+alpha/T)*np.log(alpha_k[k]) - alpha_k[k] +sum(beta*np.log(theta_k[k]))
			l3 = sum(self.n_kw(X_dk)[k]*np.log(theta_k[k]))
			LL.append(l1+l2+l3)
		return(sum(LL))

 # number of times word w assigned with topic k 
	def n_kw(self,X_dk):
		n_kw = [[0]*V for a in self.topics]
		for k in range(T):
			for w in range(V):
				c = sum([a for a in zip(*X_dk)[k]],[]).count(w)
				n_kw[k][w] = c
		return(n_kw)


	def dump(self, disp_x=False):
		if disp_x: print "X_dk:", self.X_dk
		print "n_dk:",self.n_dk
		print "pi_dk:",self.pi_dk
		print "alpha_k:",self.alpha_k
		print "theta_k:", self.theta_k
 
			











	



	

