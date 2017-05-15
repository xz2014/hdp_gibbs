#! /usr/bin/env python

#### #######################################  Gibbs sampler in HDP & Gamma-Poisson-Process
import numpy as np


class hdp_GGPP:

	def _init_(self, alpha, beta, m, V, T, corpus,sizeofstack):
		self.alpha = alpha
		self.beta = beta
		self.V = V   
		self.m = m
		self.maxNIter = maxNIter # number of iterations to update n_dk in each iteration
		self.T = T
		self.numIter = numIter # total number of iterations 
		self.sizeofstack = sizeofstack  # size of each stack, needs to be large so the empirical distribution of S_d is same as X_dk



		self.corpus = corpus
		self.D = len(corpus)
		self.topics = range(T)

		self.S_d = [np.random.choice(corpus[d],sizeofstack,replace=True).tolist() for d in range(D)] # stack for each document
		self.B_dk = [[[] for k in topics] for a in corpus] # buffer
		self.X_dk = [[[] for k in topics] for a in corpus] # each X_dk[d][k] is a list of words
		self.n_dk =[np.array([0]*T) for a in corpus] # each n_dk[d][k] is the number of words in document d with topic k

#### initialize

		self.initial_topics = [[0]*len(corpus[0])]*2
		self.alpha_k = [[] for a in topics]  
		self.theta_k = [np.zeros(V,dtype=int) for a in topics]
		self.pi_dk = [[np.ndarray(0,dtype=int) for k in topics] for a in corpus]


		for d in range(D):
			for k in range(T):
				z = [x for x in corpus[d] if initial_topics[d][k] == k ]
				X_dk[d][k].extend(z)
				n_dk[d][k] = len(z)


		




	def inference(self,numIter):
		for k in range(self.T):
			for t in range(self.numIter):
				self.parallel_for_each_topic(k)

			alpha_k = np.random.gamma(self.alpha/T+self.D,-sum(np.log(self.pi_dk)))





	def parallel_for_each_topic(self,k,maxNIter,maxBufferSize):
		n_new=[]
		for d in range(D):
			for niter in range(maxNIter):
				n_dk.append(sample_ndk(self,d,k,maxBufferSize))

		# update pi_dk for each d and k
		pi_dk = np.random.gamma(n_new[d][k]+alpha_k[k], m+1)	

		print("Topic k:")

		return(n_dk,pi_dk)







	x_ji=corpus

	t_ji=[np.zeros(len(a),dtype=int) for a in corpus] # table index
	k_jt=[[] for a in corpus]   # dish index
	n_jt = [np.ndarray(0,dtype=int) for a in corpus]  # number of words at table t in restaruant j

	tables=[[] for a in corpus] # table id
	dishes=[] # dish id

	m_k = np.ndarray(0,dtype=int)  # number of tables for each dish
	n_k = np.ndarray(0,dtype=int)  # number of customers for each dish
	n_kv = np.ndarray((0, V),dtype=int)  # number of word v for each dish 



	
# acceptance rate 
	def p_minus(self,w):
		return(n_dk[d][k]/((m*pi_dk[d][k])*theta_k[k][w]))
		

	def p_plus(self,w):
		return((m*pi_dk[d][k])/(n_dk[d][k]+1)*theta_k[k][w])


####### sample n_dk
	def sample_ndk(self, d,k, maxBufferSize):
		nacceptAdd = 0
		nacceptMinus = 0
		u = np.random.uniform(0,1)

		if u<0.5:
			x=np.random.choice(S_d[d],replace=False)
			S_d[d].remove(x)

			uu = np.random.uniform(0,1)

			if (uu < p_plus(x)): # accept with prob = min(1,p_plus)
				n_dk[d][k] =+ 1
				X_dk[d][k].extend(x)
				nacceptAdd =+ 1
			else:
				B_dk[d][k].append(x)

		else:
			if (len(X_dk[d][k]) > 0):
				x=np.random.choice(X_dk[d][k],replace=False)
				X_dk[d][k].remove(x)
				uu = np.random.uniform(0,1)
				if (uu < p_minus(x)):  
					n_dk[d][k] =- 1
					nacceptMinus =+ 1
					B_dk[d][k].append(x)
				else:
					X_dk[d][k].extend(x)
			else:
				x=np.random.choice(S_d[d],replace=False)
				S_d[d].remove(x)

				uu = np.random.uniform(0,1)

				if (uu < p_plus(x)): 
					n_dk[d][k] =+ 1
					X_dk[d][k].extend(x)
					nacceptAdd =+ 1
				else:
					B_dk[d][k].append(x)


		if (len(B_dk[d][k]) > maxBufferSize):
			S_d[d].extend(B_dk[d][k])

		return(n_dk[d][k])

# complete loglikelihood log P(n_dk, pi_dk, alpha_k, theta_k, X_dk)
	def log_likelihood(self): 
		for k in range(T):
			l1 = np.log(m)*n_dk[:][k]-sp.gammaln(alpha_k[k])+(n_dk[:][k]+alpha_k[k]-1)*np.log(pi_dk[:][k])-(m+1)*pi_dk[:][k]-np.log(factorial(n_dk[:][k]))


	def dump(self, disp_x=False):
		if disp_x: print "X_dk:", self.X_dk
		print "n_dk:",self.n_dk
		print "pi_dk:",self.pi_dk
		print "alpha_k:",self.alpha_k
		print "theta_k:", self.theta_k


	def log_factorial(self,n_dk): # log (n_dk!)
		j=0
		for i in range(n_dk):
			j=np.log(i)+j
		return(j)

 












	



	

