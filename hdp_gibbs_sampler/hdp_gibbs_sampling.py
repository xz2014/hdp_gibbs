#! /usr/bin/env python
# alpha is a k dimensional vector
# eta is fixed for all topics. v dimensional vector.
# corpus is a list of arrays, each array represent a document.

#### #######################################  Gibbs sampler
import numpy as np
from sampling_t import posterior_t

def hdp_sampling(alpha,beta,gamma,V,corpus,randomstate):
	D=len(corpus)
	x_ji=corpus

	t_ji=[np.zeros(len(a),dtype=int) for a in corpus] # table index
	k_jt=[[] for a in corpus]   # dish index
	n_jt = [np.ndarray(0,dtype=int) for a in corpus]  # number of words at table t in restaruant j

	tables=[[] for a in corpus] # table id
	dishes=[] # dish id

	m_k = np.ndarray(0,dtype=int)  # number of tables for each dish
	n_k = np.ndarray(0,dtype=int)  # number of customers for each dish
	n_kv = np.ndarray((0, V),dtype=int)  # number of word v for each dish 


## Sampling table index from conditional posterior P(t_ji | ..)
	log_likelihood=[]

	w=corpus[j][i]
	tables=tables[j]
	t_old=t_ji[j][i]
	if t_old >= 0:
		k_old = k_jt[j][t_old]

		n_kv[k_old,w] -= 1
		n_k[k_old] -= 1
		n_jt[j][t_old] -= 1

		if n_jt[j][t_old]==0:
			tables.remove(t_old)
			m_k[k_old] -= 1

			if m_k[k_old] == 0:
				dishes.remove(k_old)

	sample_t=sampling_t(j,i,w,tables)

	t_ji[j][i] = sample_t
	n_jt[j][sample_t] += 1

	sample_k = k_jt[j][sample_t]
	n_k[sample_k] += 1
	n_kv[sample_k, w] += 1

		z_update.append(sampled_topics)
		#Q=n_zw[:].tolist()
		#QQ=n_dz[:].tolist()
		#QQQ=n_z[:].tolist()
		PP.append(pp[:])
		#NZW.append(Q)
		#NDZ.append(QQ)
		L=loglikelihood_(n_zw,n_dz,alpha,eta,D,num_topics)
		log_likelihood.append(L)
		#for i in range(D):
		#	for j in range(num_topics):
		#		theta[i,j]=(QQ[i][j]+alpha[j])/(Nd[i]+sum(alpha))
		#for i in range(num_topics):
		#	for j in range(V):
		#		psi[i,j]=(Q[i][j]+eta[j])/(QQQ[i]+sum(eta))
		#theta_hat.append(theta)
		#psi_hat.append(psi)
		print('The Log-likelihood at iteration {}. is {}. Random seed{}'.format(n,L,randomstate))
############## results


#return('Topic assignments at iteration {}:{}'.format(n,sampled_topics))










