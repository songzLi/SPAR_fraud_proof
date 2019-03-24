'''
This is the prover class of SPAR.
A prover is a full node that receives LDPC coded symbols
and their Merkel proofs. It applies LDPC peeling decoding algorithm
to iteratively decode one symbol and verify its hash using one parity equation.
Once hash verification is failed, it will raise an error and prepare a
failed-parity proof.

NB: the current implementation only works for one layer.
The latest SPAR applies LDPC to every layer, which will requrie substential
extension of the current implementaiton.
'''

import numpy as np


class prover:
    def __init__(self, N: int, K: int, parity_matrix, data_root: str):
        '''
        This function initialize the prover using:
            N: number of coded symbols (data + parity)
            K: number of data symbols
            parity_matrix: an N * (N-K) binary parity matrix.
                           '1's in row-n tells the parities involving symbol-n.
                           '1's in column-p tells the symbols in parity-p.
            data_root: the top layer hash commitment of the block.
        '''
        self.N = N
        self.K = K
        self.P = N - K
        self.parity_matrix = np.array(parity_matrix)
        self.data_root = data_root

        self.symbol_values = [None] * N
        self.symbol_proofs = [None] * N
        self.symbol_parities = [np.nonzero(parity_matrix[n, :])[0]
                                for n in range(N)]

        self.parity_values = [0] * (N - K)
        self.parity_proofs = [None] * (N - K)
        self.parity_symbols = [np.nonzero(parity_matrix[:, p])[0]
                               for p in range(N - K)]
        self.parity_degree = [len(ps) for ps in self.parity_symbols]
        self.degree_1_parities = []
        self.degree_2_parities = []

        self.num_decoded_sys_symbols = 0

    def receive_symbols(self, symbols, idx, proofs):
        '''
        This function receives symbols with their Merkle proof.
        Symbols passed their proof will be used to update the decoder.
        Will return:
          - ['fraud', index of failed parity] if failed-parity
            is found in the update
          - ['rejected', an index list of the rejected received symbols]
        '''
        rejected = []
        for s, i, p in zip(symbols, idx, proofs):
            if self.symbol_hash_check(s, i, p):  # check hash
                if self.symbol_values[i] is not None:  # check whether received
                    assert self.symbol_values[i] == s  # check consistency
                else:
                    self.symbol_values[i] = s  # add to the decoder
                    out = self.decoding_update(s, i)
                    if out is not None:
                        return ['fraud', out]
            else:
                rejected.append(i)
        return ['rejected', rejected]

    def decoding_update(self, symbol, symbol_idx):
        '''
        This function unpdates the decoder by updateing the degree of
        the parities that involve the input symbol.
        If a parity's degree becomes 0, it will check whether all the symbols
        in this parity sum up to 0. If no, it will return the index of
        this parity. Otherwise, it returns None.
        '''
        if symbol_idx < self.K:  # this is an systematic symbol
            self.num_decoded_sys_symbols += 1
        # reduce the degree by one to all the parities that symbol-idx is in.
        for p in self.symbol_parities[symbol_idx]:
            self.parity_values[p] ^= symbol
            self.parity_degree[p] -= 1
            if self.parity_degree[p] == 2:
                self.degree_2_parities.append(p)
            if self.parity_degree[p] == 1:
                self.degree_1_parities.append(p)
                self.degree_2_parities.remove(p)
            if self.parity_degree[p] == 0:  # all symbols of this parity known
                if self.parity_values[p] != 0:  # fraud detection
                    return p
        return None

    def decode(self):
        '''
        When called, if not all systematic symbols are known,
        this function will perform peeling decoding until there is no degree-1
        parities.
        The function returns the decoding status and related data:
            - 'decoded': fully decoded, all hash checks passed.
                         Will return ['decoded', list of all the data symbols]
            - 'stuck': decoding stuck, need more symbols.
                      Will return ['stuck', an index list of degree-2 parities]
               get a list of symbols that will enable decoding.
            - 'fraud': found fraud.
                       Will return ['fraud', index of the failed parity].
        '''
        if self.num_decoded_sys_symbols == self.K:
            return ['decoded', self.symbol_values[:self.K]]
        while not self.degree_1_parities == []:
            for p_idx in self.degree_1_parities:
                p_value = self.parity_values[p_idx]
                # find the only unknown symbol in this parity
                for s_idx in self.parity_symbols[p_idx]:
                    if self.symbol_values[s_idx] is None:
                        # check whether this value passes hash checks
                        valid = self.decoded_symbol_check(p_value, s_idx,
                                                          p_idx)
                        if not valid:
                            return ['fraud', p_idx]
                        if valid:
                            # grant value to this symbol and update related
                            # parities
                            self.symbol_values[s_idx] = p_value
                            out = self.decoding_update(p_value, s_idx)
                            if out is not None:
                                return ['fraud', out]
                        continue  # move on to the next degree-1 parity
                self.degree_1_parities.remove(p_idx)
        if self.num_decoded_sys_symbols == self.K:
            return ['decoded', self.symbol_values[:self.K]]
        else:
            return ['stuck', self.degree_2_parities]

    def help(self):
        '''
        TBW
        This function returns an index list of symbols that will allow the
        peeling decoding to continue.
        '''
        pass

    def symbol_hash_check(self, symbol, symbol_idx, proof):
        return proof > 0

    def decoded_symbol_check(self, symbol, symbol_idx, parity_idx):
        return True


def test():

    K = 4
    N = 7
    parity_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
                              [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    data_root = 1

    prvr = prover(N, K, parity_matrix, data_root)
    assert prvr.decode() == ['stuck', []]
    symbols = [246]
    idx = [3]
    proofs = [True]
    assert prvr.receive_symbols(symbols, idx, proofs) == ['rejected', []]
    assert prvr.decode() == ['stuck', [0, 1, 2]]

    symbols = [123, 456, 135, 246]
    idx = [0, 1, 2, 3]
    proofs = [True, True, True, True]
    assert prvr.receive_symbols(symbols, idx, proofs) == ['rejected', []]
    assert prvr.decode() == ['decoded', [123, 456, 135, 246]]

    symbols = [123, 456, 135, 246]
    idx = [0, 1, 2, 3]
    proofs = [True, True, True, False]
    assert prvr.receive_symbols(symbols, idx, proofs) == ['rejected', [3]]

    prvr = prover(N, K, parity_matrix, data_root)

    symbols = [111, 111, 111, 111]
    idx = [0, 4, 5, 6]
    proofs = [True, True, True, True]
    assert prvr.receive_symbols(symbols, idx, proofs) == ['rejected', []]
    assert prvr.decode() == ['decoded', [111, 111, 111, 000]]
    print('All tests passed!')


test()
