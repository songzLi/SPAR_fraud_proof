import pickle
import numpy as np
import matplotlib.pyplot as plt
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
        self.num_decoded_symbols = 0

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
                    self.num_decoded_symbols += 1
                    if symbol_idx < self.K:
                        self.num_decoded_sys_symbols += 1
                    # print('decoded', self.num_decoded_sys_symbols)
                    symbols.append(self.parity_values[parity])
                    symbol_indices.append(symbol_idx)
            self.degree_1_parities.pop(0)
        # print('returing', symbols, symbol_indices)
        # return symbols, symbol_indices, self.num_decoded_sys_symbols == self.K
        return symbols, symbol_indices, self.num_decoded_symbols == self.N

    def symbol_update_from_reception(self, symbols, symbol_indices):
        # print('receiving symbol', symbol_indices)
        out_symbols = []
        out_indices = []
        for s, idx in zip(symbols, symbol_indices):
            if self.symbol_values[idx] is None:
                self.symbol_values[idx] == s
                self.num_decoded_symbols += 1
                if idx < self.K:
                    self.num_decoded_sys_symbols += 1
                    # print('received', self.num_decoded_sys_symbols)
                out_symbols.append(s)
                out_indices.append(idx)
        # return out_symbols, out_indices, self.num_decoded_sys_symbols == self.K
        return out_symbols, out_indices, self.num_decoded_symbols == self.N

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
            # return self.num_decoded_sys_symbols == self.K
            return self.num_decoded_symbols == self.N

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
        self.num_decoded_symbols = 0


def decoding_speed_core(dec):
    perm = np.random.permutation(dec.N)
    # perm = list(range(dec.K))
    count = 0
    # start = time.time()
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
    # duration = time.time() - start
    duration = 0

    # print('K=' + str(dec.K) + ' (' + str(int(np.sqrt(dec.K))) + ')\n',
    #       'decoded after receiving', round(count / dec.N * 100, 2),
    #       '% symbols. Took', duration, 'seconds')
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
    # print('duration:', duration)
    # print('samples:', samples)
    return np.mean(duration), np.std(duration), \
        np.mean(samples), np.std(samples)


def get_num_from_string(fileName):
    integers = [str(i) for i in range(10)]
    number = int(''.join([n for n in fileName if n in integers]))
    return number
