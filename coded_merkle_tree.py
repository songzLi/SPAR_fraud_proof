import copy
import rlp
import numpy as np

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
    med = rlp.encode(data)
    x = 1
    while x * SYMBOL_SIZE * rate < len(med):
    	x *= reduce_factor
    return med + b'\x00' * (x * SYMBOL_SIZE * rate - len(med))


def concatenation(data):
    concat = data[0]
    for i in range(1, len(data))
        concat = concat + data[i]
    return concat

# partition a byte stream into a list of hashes


def symbolPartition(data):
    return [data[i:i + 2 * HASH_SIZE] for i in range(0, len(data), 2 * HASH_SIZE)]


def LDPC_encoding(symbols, rate):


def hashAggregate(coded_symbols):
	hashes = [sha3(x) for x in coded_symbols]
    # N is number of hashes to be aggregated
    N = len(hashes)
    # aggregate hashes of systematic symbols 
    systematic = [concatenation(hashes[i: i + C*rate]) for i in range(0, N*rate, C*rate)]
    # aggregate hashes of parity symbols 
    parity = [concatenation(hashes[i: i + C*(1-rate)]) for i in range(N*rate, N, C*(1-rate))]

    assert len(systematic)== len(parity)

    return [systematic[i] + parity[i] for i in range(0, len(systematic))]

def nextIndex(index, K):
    if index <= K-1:  #this symbol is a systematic symbol
        newIndex = index//reduce_factor
    else: #this symbol is a parity symbol
        newIndex = (index-K)//(C-reduce_factor)
    return newIndex


class Coded_merkle_tree:
    def __init__(self, data, headerSize): #headerSize is measured as number of hashes stored in the header for the constructed coded merkle tree
    	pdata = pad(data)
        # partition the transaction block into symbols of SYMBOL_SIZE bytes
        # here each symbol is an array of bytes
    	symbols = [concatenation(pdata[i: i + SYMBOL_SIZE]) for i in range(0, len(pdata), SYMBOL_SIZE)]
        # Create coded symbols using selected LDPC code
    	coded_symbols = LDPC_encoding(symbols,rate)
        # N denotes the number of coded symbols in the original coded block
        self.N = len(coded_symbols)
        # compute number of levels in the coded merkle tree given intended header size
        level = np.log(self.N/headerSize)//np.log(reduce_factor) +1
        # Construct a coded merkle tree with level layers, the first layer is the original data block encoded by LDPC code
    	self.tree = [coded_symbols]
    	for j in range(0,level-1):
    		symbols = hashAggregate(tree[j])
    		self.tree.append(LDPC_encoding(symbols,rate))
        # roots are the hashes of the last layer, and we will store roots in the block header    
    	self.roots = [sha3(x) for x in self.tree[level-1]]


    # Make a Merkle proof for some index
    # A proof for a particular symbol is a list of symbols in the upper levels
    def proof(self, index):
        assert 0 <= index < self.N
        merkle_proof = list()
        moving_index = index # index of a symbol in the proof list for its level
        moving_k = self.N * rate # number of systematic symbols in a level
        for i in range(len(self.tree)-1):
            moving_index = nextIndex(moving_index,moving_k)
            merkle_proof.append(self.tree[i+1][moving_index])
            moving_k = moving_k/reduce_factor
        return merkle_proof

# verify each symbol in the list matches its hash
def verify_proof(index, symbol, K, proof, roots):
    current_index = index
    current_symbol = symbol
    current_k = K
    for s in proof:
        # recover the hash values from the symbol s
        h = symbolPartition(s)
        if current_index <= current_k-1: #this symbol is a systematic symbol
            hashIndex = current_index % reduce_factor #find the index of the hash of this systematic symbol 
        else: #this symbol is a parity symbol
            hashIndex = (current_index-current_k) % (C-reduce_factor) + reduce_factor #find the index of the hash of this parity symbol 

        if sha3(current_symbol) != h[hashIndex]: #hash check
            return False
        else:
            current_index = nextIndex(current_index, current_k):
            current_symbol = s
            current_k = current_k/reduce_factor

    # final check against the root hashes stored in the header        
    if sha3(current_symbol) == roots[current_index]:
        return True
    else:
        return False










        













