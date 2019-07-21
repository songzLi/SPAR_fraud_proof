import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
# import copy

SYMBOL = 0


class decoder:
    def __init__(self, file_name):
        with open(file_name, 'rb') as handle:
            out = pickle.load(handle)
        self.parities = out['parities']
        self.symbols = out['symbols']
        # self.parities = [[0, 1, 3, 4], [1, 2, 3, 5], [0, 2, 3, 6]]
        # self.symbols = [[0, 2], [0, 1], [1, 2], [0, 1, 2], [0], [1], [2]]
        self.N = len(self.symbols)
        self.P = len(self.parities)
        self.K = self.N - self.P
        # print('K=', self.K, 'N=', self.N)
        self.symbol_values = [None] * self.N
        self.parity_values = [SYMBOL] * (self.N - self.K)
        self.parity_proofs = [None] * (self.N - self.K)
        self.parity_degree = [len(ps) for ps in self.parities]
        self.degree_1_parities = []
        self.degree_2_parities = []

        self.num_decoded_sys_symbols = 0

    def parity_update(self, symbols, symbol_indices):
        # print('updating parities related to', symbol_indices)
        if len(symbols) == 0:
            has_degree_1_parities = self.degree_1_parities is not []
            return has_degree_1_parities
        for s, idx in zip(symbols, symbol_indices):
            # parity_list = copy.deepcopy(self.symbols[idx])
            parity_list = self.symbols[idx][:]
            # print('symbol', idx, 'is connected to parities', parity_list)
            # print(s, idx)
            for parity in parity_list:
                self.parity_values[parity] ^= s
                self.parity_degree[parity] -= 1
                self.parities[parity].remove(idx)
                self.symbols[idx].remove(parity)
                if self.parity_degree[parity] == 1:
                    self.degree_1_parities.append(parity)
                    # print('adding parity', parity, self.parities[parity])
        has_degree_1_parities = self.degree_1_parities is not []
        return has_degree_1_parities

    def symbol_update_from_degree_1_parities(self):
        # print('updating')
        symbols = []
        symbol_indices = []
        # print(self.degree_1_parities)
        for parity in self.degree_1_parities[:]:
            # print('updating parity', parity, self.parities[parity])
            if not self.parities[parity] == []:
                symbol_idx = self.parities[parity][0]
                if self.symbol_values[symbol_idx] is None:
                    self.symbol_values[symbol_idx] = \
                        self.parity_values[parity]
                    if symbol_idx < self.K:
                        self.num_decoded_sys_symbols += 1
                    # print('decoded', self.num_decoded_sys_symbols)
                    symbols.append(self.parity_values[parity])
                    symbol_indices.append(symbol_idx)
            self.degree_1_parities.pop(0)
        # print('returing', symbols, symbol_indices)
        return symbols, symbol_indices, self.num_decoded_sys_symbols == self.K

    def symbol_update_from_reception(self, symbols, symbol_indices):
        # print('receiving symbol', symbol_indices)
        out_symbols = []
        out_indices = []
        for s, idx in zip(symbols, symbol_indices):
            if self.symbol_values[idx] is None:
                self.symbol_values[idx] == s
                if idx < self.K:
                    self.num_decoded_sys_symbols += 1
                    # print('received', self.num_decoded_sys_symbols)
                out_symbols.append(s)
                out_indices.append(idx)
        return out_symbols, out_indices, self.num_decoded_sys_symbols == self.K

    def peeling_decode(self):
        # print('peeling')
        while True:
            # tim
            symbols, symbol_indices, decoded = \
                self.symbol_update_from_degree_1_parities()
            if decoded:
                return decoded
            if not symbols == []:
                # print('continue peeing with', symbols)
                keep_peeling = self.parity_update(symbols, symbol_indices)
                if keep_peeling:
                    continue
            return self.num_decoded_sys_symbols == self.K

    def reset(self):
        # reset the decoder, so that it can be reused without loading the
        # parity matrix again
        self.symbol_values = [None] * self.N
        self.parity_values = [SYMBOL] * (self.N - self.K)
        self.parity_proofs = [None] * (self.N - self.K)
        self.parity_degree = [len(ps) for ps in self.parities]
        self.degree_1_parities = []
        self.degree_2_parities = []

        self.num_decoded_sys_symbols = 0


def decoding_speed_core(dec):
    perm = np.random.permutation(dec.N)
    # perm = list(range(dec.K))
    count = 0
    start = time.time()
    while True:
        # time.sleep(1)
        symbols, symbol_indices, decoded = \
            dec.symbol_update_from_reception([SYMBOL], [perm[count]])
        if decoded:
            break
        count += 1
        if dec.parity_update(symbols, symbol_indices):
            decoded = dec.peeling_decode()
            if decoded:
                break
    duration = time.time() - start

    print('K=' + str(dec.K) + ' (' + str(int(np.sqrt(dec.K))) + ')\n',
          'decoded after receiving', round(count / dec.N * 100, 2),
          '% symbols. Took', duration, 'seconds')
    return duration, count / dec.N * 100


def decoding_speed_wrap(file_name, num_iters):
    duration = []
    samples = []
    dec = decoder(file_name)
    for i in range(num_iters):
        # print('iteration', i)
        dec.reset()
        d, s = decoding_speed_core(dec)
        duration += [d]
        samples += [s]
    print('duration:', duration)
    print('samples:', samples)
    return np.mean(duration), np.std(duration), \
        np.mean(samples), np.std(samples)


def get_num_from_string(fileName):
    integers = [str(i) for i in range(10)]
    number = int(''.join([n for n in fileName if n in integers]))
    return number


def run():
    files = [f for f in listdir('./')
             if isfile(join('./', f)) and f[-7:] == '.pickle' and
             f[:7] == 'symbols']
    K_group = np.sort([get_num_from_string(f) for f in files])
    print('K_group:', K_group)
    print('K_sqrt:', [np.sqrt(K) for K in K_group])
    result = {}
    result['time'] = []
    result['time_std'] = []
    result['count'] = []
    result['count_std'] = []
    result['K_group'] = K_group
    num_iters = 20
    for K in K_group:
        file_name = 'symbols_and_parities_' + str(K) + '.pickle'
        t, t_std, pct, pct_std = decoding_speed_wrap(file_name, num_iters)
        result['time'].append(t)
        result['time_std'].append(t_std)
        result['count'].append(pct)
        result['count_std'].append(pct_std)

        with open('spar_performance_' + str(K_group[0]) +
                  '_' + str(K_group[-1]) +
                  '.pickle', 'wb') as handle:
            pickle.dump(result, handle)

    with open('spar_performance_' + str(K_group[0]) +
              '_' + str(K_group[-1]) +
              '.pickle', 'rb') as handle:
        result = pickle.load(handle)

    k_spar = result['K_group']
    d_spar = result['time']
    pct_spar = result['count']
    plt.loglog(k_spar, d_spar,
               label='SPAR', linewidth=3, marker='s', markersize=8,
               markerfacecolor='w')
    for k, dt in zip(k_spar, d_spar):
        plt.text(k * 0.95, dt * 1.1,
                 str(int(k * 256 * 4 / 1024 / 1024)) + 'MB')

    plt.xlabel('K')
    plt.ylabel('decoding time (second)')
    plt.grid()
    plt.title('decoding time')
    plt.tight_layout()
    plt.legend()
    plt.show()

    plt.plot(K_group, pct_spar)
    plt.title('percent to decode')
    plt.show()


# run()
