import poly_utils as pu
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle


def decoding_time_2drs(K, R):
    total_time = 0
    num_rows = K

    for n in range(num_rows):
        # print('decoding row-' + str(n))
        x = np.random.randint(0, 65535, K)
        idx = np.random.permutation(range(int(K / R)))[:K]
        ts = time.time()
        pu.lagrange_interp(x, idx)
        te = time.time()
        total_time += (te - ts)
    print('decoding time:', total_time, 'seconds')
    return total_time


# dt = []
# R = 0.5
# K_group = [128 * i for i in range(1, 11)]
# result = {}
# result['K_group'] = K_group
# for K in K_group:
#     print(K)
#     dt.append(decoding_time_2drs(K, R))
#     result['decoding_time'] = dt
#     with open('RS_decoding.pickle', 'wb') as handle:
#         pickle.dump(result, handle)

with open('RS_decoding.pickle', 'rb') as handle:
    result = pickle.load(handle)

plt.semilogy(result['K_group'], result['decoding_time'], label='2D-RS',
             linewidth=3,
             marker='o', markersize=8,
             markerfacecolor='w')
# plt.semilogy([128, 256, 384], [6.8, 77.6, 494], label='spar')
for k, dt in zip(result['K_group'], result['decoding_time']):
    plt.text(k * 0.95, dt * 1.1,
             str(int(k ** 2 * 256 * 4 / 1024 / 1024)) + 'MB')
plt.xlabel('$\sqrt{K}$')
plt.ylabel('decoding time (second)')
plt.grid()
plt.title('2D-RS block decoding time using $GF_{65536}$' +
          ' when $\sqrt{K}>128$\n' +
          '(Lagrange interpolation on size-$\sqrt{K}$ rows for $\sqrt{K}$ times)')
plt.tight_layout()
plt.show()
