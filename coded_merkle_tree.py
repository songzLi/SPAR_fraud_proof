import copy
import rlp
import numpy as np
import random
import os
import string
import copy

try:
    from Crypto.Hash import keccak
    sha3 = lambda x: keccak.new(digest_bits=256, data=x).digest()
except ImportError:
    import sha3 as _sha3
    sha3 = lambda x: _sha3.sha3_256(x).digest()

# A hash has 32 bytes
HASH_SIZE = 32

# We choose each symbol of the original block to have 256 bytes
SYMBOL_SIZE = 256

# number of hashes to aggregate to form a new symbol
C = 8

# Coding rate
rate = 0.25

# number of symbols reduces by reduce_factor each layer upward
reduce_factor = C * rate


# Returns the smallest power of reduce_factor equal to or greater than a number
# def higher_power_of_2(x):
#     higher_power_of_2 = 1
#     while higher_power_of_2 < x:
#         higher_power_of_2 *= 2
#     return higher_power_of_2

def pad(data):
    #med = rlp.encode(data) #using rlp encoding from Ethereum
    med = data
    #print(med)
    #print(len(med))
    x = 1
    while x * SYMBOL_SIZE * rate < len(med):
        x *= reduce_factor
    return med + b'\x00' * (int(x * SYMBOL_SIZE * rate) - len(med))


def concatenation(data):
    concat = data[0]
    for i in range(1, len(data)):
        concat = concat + data[i]
    return concat

# partition a byte stream into a list of hashes


def symbolPartition(data):
    return [data[i:i + HASH_SIZE] for i in range(0, len(data), HASH_SIZE)]


def LDPC_encoding(symbols, rate):
    coded_symbols = symbols 
    for i in range(int((1/rate-1)*len(symbols))):
        coded_symbols.append(b'\x00' * SYMBOL_SIZE)
    return coded_symbols


def hashAggregate(coded_symbols):
    hashes = [sha3(x) for x in coded_symbols]
    #print(hashes)
    #print(len(hashes[0]))
    # N is number of hashes to be aggregated
    N = len(hashes)
    # aggregate hashes of systematic symbols
    systematic = [concatenation(hashes[i: i + int(C * rate)])
                  for i in range(0, int(N * rate), int(C * rate))]
    # aggregate hashes of parity symbols
    parity = [concatenation(hashes[i: i + int(C * (1 - rate))])
              for i in range(int(N * rate), N, int(C * (1 - rate)))]

    assert len(systematic) == len(parity)

    # print(systematic)
    # print(len(systematic))
    # print(parity)
    # print(len(parity))

    # print([systematic[i] + parity[i] for i in range(0, len(systematic))])

    return [systematic[i] + parity[i] for i in range(0, len(systematic))]


def nextIndex(index, K):
    if index <= K - 1:  # this symbol is a systematic symbol
        newIndex = index // reduce_factor
    else:  # this symbol is a parity symbol
        newIndex = (index - K) // (C - reduce_factor)
    return int(newIndex)

class coded_merkle_tree:

# headerSize is measured as number of hashes stored in the header for the
# constructed coded merkle tree
    def __init__(self, data, headerSize):
        pdata = pad(data)
        #print(len(pdata))

        # partition the transaction block into symbols of SYMBOL_SIZE bytes
        # here each symbol is an array of bytes
        symbols = [pdata[i: i + SYMBOL_SIZE]
                   for i in range(0, len(pdata), SYMBOL_SIZE)]
        #print(symbols)
        #print(len(symbols))

        # Create coded symbols using selected LDPC code
        coded_symbols = LDPC_encoding(symbols, rate)
        # N denotes the number of coded symbols in the original coded block
        self.N = len(coded_symbols)
        # compute number of levels in the coded merkle tree given intended
        # header size
        level = np.log(self.N / headerSize) // np.log(reduce_factor) + 1
        # print(level)
        # Construct a coded merkle tree with level layers, the first layer is
        # the original data block encoded by LDPC code
        self.tree = [coded_symbols]
        for j in range(0, int(level) - 1):
            symbols = hashAggregate(self.tree[j])
            self.tree.append(LDPC_encoding(symbols, rate))
        # roots are the hashes of the last layer, and we will store roots in
        # the block header
        self.roots = [sha3(x) for x in self.tree[int(level) - 1]]

    def hide_symbols(self, hidePattern):
        # hidePattern is a binary array indicating the availablity of the coded
        # symbols
        self.hide = hidePattern

    # Return a Merkle proof for some symbol index at some level of the coded merkle tree
    # A proof for a particular symbol is a list of symbols in the upper levels
    def proof(self, lvl, index):
        assert 0 <= index < len(self.tree[lvl])
        merkle_proof = list()
        moving_index = index  # index of a symbol in the proof list for its level
        moving_k = int(len(self.tree[lvl]) * rate) # number of systematic symbols in a level
        for i in range(lvl,len(self.tree) - 1):
            moving_index = int(nextIndex(moving_index, moving_k))
            merkle_proof.append(self.tree[i + 1][moving_index])
            moving_k = moving_k // reduce_factor
        return merkle_proof

    # Return the requested symbols together with their merkle proofs
    def sampling(self, lvl, S):  # S is the list of indices required by the light client at level lvl
        symbols = list()
        for i in S:
            if self.hide[lvl][i] == 0:  # requested symbol was not hided
                symbols.append((self.tree[lvl][i], self.proof(lvl,i)))
            else:
                print('Requested symbol at level {} with index {} is not available.'.format(lvl,i))
                return []
        return symbols


# verify each symbol in the list matches its hash
def verify_proof(lvl, index, symbol, proof, K, roots):
    current_index = index
    current_symbol = symbol
    current_lvl = lvl
    current_k = K//(reduce_factor**lvl)
    for s in proof:
        # recover the hash values from the symbol s
        h = symbolPartition(s)
        if current_index <= current_k - 1:  # this symbol is a systematic symbol
            # find the index of the hash of this systematic symbol
            hashIndex = int(current_index % reduce_factor)
        else:  # this symbol is a parity symbol
            # find the index of the hash of this parity symbol
            hashIndex = int((current_index - current_k) % (C -
                                                       reduce_factor) + reduce_factor)

        if sha3(current_symbol) != h[hashIndex]:  # hash check
            #print(sha3(current_symbol))
            #print(h[hashIndex])
            print('Failed at level {} with symbol index {}.'.format(current_lvl,current_index))
            return False
        else:
            current_index = nextIndex(current_index, current_k)
            current_symbol = s
            current_k = int(current_k // reduce_factor)
            current_lvl = current_lvl + 1

    # final check against the root hashes stored in the header
    if sha3(current_symbol) == roots[current_index]:
        return True
    else:
        #print('I have reach the last level.')
        return False


# test merkle tree construction
lettersAndDigits = string.ascii_letters + string.digits
data = ''.join(random.choice(lettersAndDigits) for i in range(10**4)).encode()
#print(data)
#print(len(data))
#print(data[0])
#data[0:256]
header_size = 8  # header contains 8 hashes
codedMerkleTree = coded_merkle_tree(data, header_size)
assert len(codedMerkleTree.roots) == header_size
block_length = codedMerkleTree.N
K = int(block_length*rate)
print('The constructed coded merkle tree has {} levels.'.format(len(codedMerkleTree.tree)))
#print(codedMerkleTree.tree[0][1])


#test merkle proof generated at certain level for a symbol index
proof = codedMerkleTree.proof(3, 4)
print(proof)
assert len(codedMerkleTree.tree)-1 - 3 == len(proof) 

#test symbol sampling function
levels = len(codedMerkleTree.tree)
hide = list()
for i in range(levels):
    hide.append([0 for x in range(len(codedMerkleTree.tree[i]))])
#print(hide)
codedMerkleTree.hide_symbols(hide)
#print(codedMerkleTree.tree[0][1])
print(codedMerkleTree.sampling(0, [9, 10]))
hide[0][1] = 1
print(codedMerkleTree.sampling(0, [1]))

#test proof verification at honest full node
symbol = codedMerkleTree.tree[3][4]
roots = codedMerkleTree.roots
print(verify_proof(3, 4, symbol, proof, K, roots))

symbol = codedMerkleTree.tree[0][15]
proof = codedMerkleTree.proof(0, 15)
roots = codedMerkleTree.roots
print(verify_proof(0, 15, symbol, proof, K, roots))










