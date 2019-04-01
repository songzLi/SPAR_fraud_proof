import numpy as np
import matplotlib.pyplot as plt

p_2drs = 0.25
p_spar = 0.124

sample_size = range(1, 81)
miss_prob_2drs = [(1 - p_2drs) ** s for s in sample_size]
miss_prob_spar = [(1 - p_spar) ** s for s in sample_size]
plt.semilogy(sample_size, miss_prob_2drs, label='2DRS')
plt.semilogy(sample_size, miss_prob_spar, label='SPAR')
plt.xlabel('number of samples')
plt.ylabel('sampling failure probability')
plt.legend()
plt.grid()
plt.show()
