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
		self.m = m  # n_dk is poisson ~ ( m * pi_dk ) , empirically set as 1/T. 
		self.T = T   # number of topics
		self.corpus = corpus
		self.sizeofstack = sizeofstack # size of each stack for a document. 
		self.numIter = numIter
		self.maxNIter = maxNIter # maxNIter: number of iterations to update n_dk in each iteration
		self.maxBufferSize = maxBufferSize   # if length of B_dk exceeds maxBufferSize, push all the elements from B_dk to S_d

		self.D = len(corpus)
		self.topics = range(T)  # topics are indexed as 0,1,2...
		self.initial_topics = [[0]*len(corpus[0])]*2  # initial topic assignment
		# memoization
		self.aT = alpha*1.0/T

##### All variables:
		self.S_d = [np.random.choice(corpus[d],sizeofstack,replace=True).tolist() for d in range(self.D)] # the stack for each document
		self.B_dk = [[[] for k in self.topics] for a in corpus] # buffer
		self.X_dk = [[[] for k in self.topics] for a in corpus] # each X_dk[d][k] is a list of words
		self.n_dk =[[0]*T for a in corpus] # each n_dk[d][k] is the number of words in document d with topic k
		self.alpha_k = [1]*T
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
		self.m = 1*1.0/self.T




######### for a given k, run gibbs sampling 

	def parallel_samplging_for_each_topic(self,k): 
		self.initial_()
		new_n_with_topic_k=[]
	#	L=[]

		for t in range(1,self.numIter):
			for d in range(self.D):
				# update n_dk for each d and k
				self.n_dk[d][k] = self.updating_ndk(d, k)
				
				# update pi_dk for each d and k
				self.pi_dk[d][k] = np.random.gamma(self.n_dk[d][k]+self.alpha_k[k], self.m+1)


			# update alpha_k for each k
			self.alpha_k[k] = self.sample_alpha_k(k)

			# update theta_k for each k
			prob_theta_k = np.exp(sum((map(lambda x,y:x+y, self.beta,self.n_kw(self.X_dk)[k]))*np.log(self.theta_k[k])))
			self.theta_k[k] = np.random.dirichlet(np.add(self.beta,self.n_kw(self.X_dk)[k]))

			# compute log likelihood
			l = self.log_likelihood()
	#		L.append(l)

			print "The Log-likelihood at iteration {} is {}. Topic{}".format(t,l,k)

	#	return(L)


	

#### for a given d and k, update n_dk using metroplis hasting algorithm. maxNIter iterations

	def updating_ndk(self, d, k):
		for iter in range(1,self.maxNIter):
			newn =self.sample_ndk(d, k)
		return(newn)


####### sample n_dk, one iteration
	def sample_ndk(self, d,k):
		u = np.random.uniform(0,1)

		if u<0.5:
			x=np.random.choice(self.S_d[d],replace=False)
			self.S_d[d].remove(x)

			uu = np.random.uniform(0,1)

			if (uu < self.p_plus(d,k,x)): # accept with prob = min(1,p_plus)
				self.n_dk[d][k] += 1
				self.X_dk[d][k].append(x)
			else:
				self.B_dk[d][k].append(x)

		else:
			if (len(self.X_dk[d][k]) > 0):
				x=np.random.choice(self.X_dk[d][k],replace=False)
				self.X_dk[d][k].remove(x)
				uu = np.random.uniform(0,1)
				if (uu < self.p_minus(d,k,x)):  
					self.n_dk[d][k] -= 1
					self.B_dk[d][k].append(x)
				else:
					self.X_dk[d][k].append(x)

		if (len(self.B_dk[d][k]) > self.maxBufferSize):
			self.S_d[d].extend(self.B_dk[d][k])

		return(self.n_dk[d][k])




# acceptance rate 
	def p_minus(self,d,k,w):
		if  self.pi_dk[d][k]*self.theta_k[k][w] == 0 :
			return (1.0)
		else:
			return(1.0*self.n_dk[d][k]/(self.m*self.pi_dk[d][k]*self.theta_k[k][w]))

	def p_plus(self,d,k,w):
		return((self.m*self.pi_dk[d][k])*self.theta_k[k][w]/(self.n_dk[d][k]+1))

	def sample_alpha_k(self,k):
		if -sum(np.log(zip(*self.pi_dk)[k])) > 0:
			return(np.random.gamma(self.aT+self.D,-sum(np.log(zip(*self.pi_dk)[k]))))
		else:
			return(self.alpha_k[k])


# complete loglikelihood log P(n_dk, pi_dk, alpha_k, theta_k, X_dk)
	def log_likelihood(self):
		LL=[]
		for k in range(self.T):
			v = np.array(zip(*self.n_dk)[k])
			q = np.array(zip(*self.pi_dk)[k])
			e = np.array(zip(*self.X_dk)[k])
			l1 = sum(np.log(self.m)*v-sp.gammaln(self.alpha_k[k])+(v+self.alpha_k[k]-1)*np.log(q)-(self.m+1)*q-np.log(sm.factorial(v)))
			l2 = (-1+self.aT)*np.log(self.alpha_k[k]) - self.alpha_k[k] +sum(self.beta*np.log(self.theta_k[k]))
			l3 = sum(self.n_kw(self.X_dk)[k]*np.log(self.theta_k[k]))
			LL.append(l1+l2+l3)
		return(sum(LL))

 # number of times word w assigned with topic k 
	def n_kw(self,X_dk):
		n_kw = [[0]*self.V for a in self.topics]
		for k in range(self.T):
			for w in range(self.V):
				c = sum([a for a in zip(*self.X_dk)[k]],[]).count(w)
				n_kw[k][w] = c
		return(n_kw)


	def dump(self, disp_x=False):
		if disp_x: print "X_dk:", self.X_dk
		print "n_dk:",self.n_dk
		print "pi_dk:",self.pi_dk
		print "alpha_k:",self.alpha_k
		print "theta_k:", self.theta_k
 
			











	



	

