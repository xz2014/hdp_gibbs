import numpy as np
from import new_table
def posterior_t(j,i,w,tables):
	f_k = (n_kv[:,w]+beta)/(V*beta+n_k)
	p_xji = (sum(f_k*(m_k))+gamma/V)/(gamma+sum(m_k))
	p_t = [n_jt[j][t] * f_k[k_jt[j][t]] for t in tables]
	p_t.append(alpha*p_xji)

	p_t.np.array(p_t,copy=False)
	p_t /= p_t.sum()

	tt=np.random.multinomial(1, p_t).argmax()

	if tt < len(tables):
		return tables[tt]
	else:
		return new_table(j, i, f_k)

