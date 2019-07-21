import ldpc_decoder as lds
import numpy as np
# import time
import matplotlib.pyplot as plt
# from os import listdir
# from os.path import isfile, join
import pickle


def decoding_ratio_core(K, num_iters=1000000):
    r = np.zeros(num_iters)
    file_name = 'symbols_and_parities_' + str(K) + '.pickle'
    # for n in range(num_iters):
    #     _, _, pct, pct_std = lds.decoding_speed_wrap(file_name, 1)
    #     r[n] = pct
    #     if n % 10 == 0:
    #         print(n, '/', num_iters)
    #         with open('spar_decoding_ratio_' + str(K) +
    #                   '.pickle', 'wb') as handle:
    #             pickle.dump(r, handle)

    with open('spar_decoding_ratio_' + str(K) + '.pickle', 'rb') as handle:
        r = pickle.load(handle)
    num_iters = len(r)
    plt.semilogy(np.sort(r), [1 - n / num_iters for n in range(num_iters)])
    plt.xlabel('% of symbols received', fontsize=14)
    plt.ylabel('prob. undecodable after receiving x% symbols', fontsize=14)
    plt.xlim((50, 60))
    plt.grid()
    # plt.title('CCDF of decoding ratio when K=' + str(K))
    plt.tight_layout()
    plt.legend(fontsize=14)
    plt.show()


decoding_ratio_core(K=4096, num_iters=100000)
