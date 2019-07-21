import numpy as np
import pickle
from matplotlib import pyplot as plt


class rs_decoder:
    def __init__(self, K, R=0.25):
        self.K = K
        self.N = int(K / R)
        self.KS = int(np.sqrt(self.K))
        self.NS = int(np.sqrt(self.N))
        self.row_degree = [0] * self.NS
        self.column_degree = [0] * self.NS
        self.decoded_orign = 0
        self.matrix = np.zeros([self.NS, self.NS])
        self.received_symbols = []
        self.full_rows_and_cols = []

    def decode(self):
        return 1

    def receive_symbol(self, values, positions):
        # print('receiving symbols')
        symbols = []
        pos = []
        for value, position in zip(values, positions):
            x = position[0]
            y = position[1]
        # print('receiving', value, 'at', position)
        if self.matrix[x, y] == 0:
            # print('useless:', x, y, self.matrix[x, y])
            self.matrix[x, y] = value
            # print('receiving', value, 'at', position)
            symbols.append(value)
            pos.append(position)
            if x < self.KS and y < self.KS:
                self.decoded_orign += 1
        return symbols, pos, self.decoded_orign == self.K

    def update_row_and_columns(self, symbols, positions):
        # self.full_rows.remove(x)
        for position in positions:
            x = position[0]
            y = position[1]
            self.row_degree[x] += 1
            self.column_degree[y] += 1
            if self.row_degree[x] == self.KS:
                self.full_rows_and_cols.append([0, x])
            if self.column_degree[y] == self.KS and y < self.KS:
                self.full_rows_and_cols.append([1, y])
        return not self.full_rows_and_cols == []

    def update_symbols_from_rows_and_columns(self):
        # self.full_columns.remove(y)
        # print('decoded', idx_list, 'at column', y)
        # print('updating symbols')
        symbols = []
        pos = []
        for array in self.full_rows_and_cols[:]:
            if array[0] == 0:
                x = array[1]
                for i in range(self.NS):
                    if self.matrix[x, i] == 0:
                        self.matrix[x, i] = self.decode()
                        symbols.append(self.matrix[x, i])
                        # print('decoded', [x, i])
                        # print(self.matrix)
                        # print('total decoded origin', self.decoded_orign)
                        pos.append([x, i])
                        if x < self.KS and i < self.KS:
                            self.decoded_orign += 1
            if array[0] == 1:
                y = array[1]
                if y < self.KS:
                    for i in range(self.NS):
                        if self.matrix[i, y] == 0:
                            self.matrix[i, y] = self.decode()
                            # print('decoded', [i, y])
                            # print(self.matrix)
                            symbols.append(self.matrix[i, y])
                            pos.append([i, y])
                            if i < self.KS:
                                self.decoded_orign += 1
            self.full_rows_and_cols.pop(0)
        return symbols, pos, self.decoded_orign == self.K

    def peeling_decode(self):
        while True:
            # print('calling update_symbols inside peeling_decode()')
            symbols, positions, decoded = \
                self.update_symbols_from_rows_and_columns()
            if decoded:
                return decoded
            if not symbols == []:
                # print('calling update_row inside peeling_decode()')
                keep_peeling = self.update_row_and_columns(symbols, positions)
                if not keep_peeling:
                    return self.decoded_orign == self.K

    def receive_symbol_wrap(self, symbols, positions):
        # print('calling receive_symbols inside receive_symbol_wrap()')
        symbols, positions, decoded = self.receive_symbol(symbols, positions)
        if decoded:
            return decoded
        # print('calling update_row inside receive_symbol_wrap()')
        if self.update_row_and_columns(symbols, positions):
            # print('calling peeling_decode inside receive_symbol_wrap()')
            return self.peeling_decode()


def run_core(K, R=0.25):
    N = int(K / R)
    NS = int(np.sqrt(N))
    perm = np.random.permutation(N)
    position = [[int(p / NS), p % NS] for p in perm]
    rs = rs_decoder(K, R)
    count = 0
    while True:
        decoded = rs.receive_symbol_wrap([1], [position[count]])
        count += 1
        if decoded:
            break
    print('decoded after receiving', count / N * 100, '% symbols')
    return count


def decoding_ratio_core(K, num_iters=1000, R=0.25):
    r = np.zeros(num_iters)
    N = K / R
    for i in range(num_iters):
        print(i)
        r[i] = run_core(K, R) / N * 100
    with open('2DRS_decoding_ratio_K=' + str(K) + '.pickle', 'wb') as handle:
        pickle.dump(r, handle)
    with open('2DRS_decoding_ratio_K=' + str(K) + '.pickle', 'rb') as handle:
        r = pickle.load(handle)

    plt.semilogy(np.sort(r), [1 - n / num_iters for n in range(num_iters)])
    plt.xlabel('% of symbols received')
    plt.ylabel('prob. undecodable after receiving x% symbols', fontsize=14)
    plt.xlim((0, 100))
    plt.grid()
    plt.title('CCDF of decoding ratio when K=' + str(K))
    plt.tight_layout()
    plt.legend()
    plt.show()


decoding_ratio_core(K=4096, num_iters=1000000)
