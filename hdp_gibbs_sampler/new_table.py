import numpy as np
from sampling_dish import sampling_dish
def new_table(j,i,f_k):
	T_j = n_jt[j].size
	for t_new in range(T_j):
		if t_new not in tables[j]: break
	else:
		t_new=T_j # create a new table id
		n_jt[j].resize(t_new+1)
		n_jt[j][t_new] = 0 # initialize new table count
		k_jt[j].append(0)

	tables[j].append(t_new)

	#sampling a dish for new table
	p_k=[ m_k[a]*f_k[a] for a in len(f_k)]
	p_k.append(gamma/V)
	k_new = sampling_dish(np.array(p_k),copy=False)

	k_jt[j][t_new] = k_new
	m_k[k_new] += 1

	return t_new

