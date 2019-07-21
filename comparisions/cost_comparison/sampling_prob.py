import numpy as np
import matplotlib.pyplot as plt

p_2drs = 0.25
p_spar_strong = 0.125
p_spar_weak = 0.43

sample_size = range(1, 51)
miss_prob_2drs = [(1 - p_2drs) ** s for s in sample_size]
miss_prob_spar_strong = [(1 - p_spar_strong) ** s for s in sample_size]
miss_prob_spar_weak = [(1 - p_spar_weak) ** s for s in sample_size]
plt.semilogy(sample_size, miss_prob_2drs, label='2DRS', color='orange')
plt.semilogy(sample_size, miss_prob_spar_strong,
             label='SPAR (strong adversary)', color='b')
plt.semilogy(sample_size, miss_prob_spar_weak,
             label='SPAR (weak adversary)', color='g')
plt.xlabel('number of samples', fontsize=16)
plt.ylabel('sampling failure probability', fontsize=16)
plt.legend(fontsize=14, loc='best')
plt.grid()
plt.ylim((1e-6, None))
plt.tight_layout()
plt.show()
