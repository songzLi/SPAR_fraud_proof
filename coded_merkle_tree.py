import copy
import rlp

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

#Coding rate
rate = 0.5

#number of symbols reduces by reduce_factor each layer upward
reduce_factor = C * rate


# Returns the smallest power of reduce_factor equal to or greater than a number
def higher_power_of_2(x):
    higher_power_of_2 = 1
    while higher_power_of_2 < x:
        higher_power_of_2 *= 2
    return higher_power_of_2

def pad(data):
    med = rlp.encode(data)
    x = 1
    while x * SYMBOL_SIZE * rate < len(med):
    	x *= reduce_factor
    return med + b'\x00' * (x * SYMBOL_SIZE * rate - len(med))

def LDPC_encoding(symbols,rate):


def hashAggregate(coded_symbols):
	hashes = [sha3(x) for x in coded_symbols]
	symbols = [hashes[i: i + C] for i in range(0, len(hashes), HASH_SIZE)]


class Coded_merkle_tree:
    def __init__(self, data, level):
    	pdata = pad(data)
    	symbols = [pdata[i: i + SYMBOL_SIZE] for i in range(0, len(pdata), SYMBOL_SIZE)]
    	coded_symbols = LDPC_encoding(symbols,rate)
        # Construct a coded merkle tree with level layers, the first layer is the original data block encoded by LDPC code
    	tree = [coded_symbols]
    	for j in range(0,level-1):
    		symbols = hashAggregate(tree[j])
    		tree.append(LDPC_encoding(symbols,rate))
        # roots are the hashes of the last layer, and we will store roots in the block header    
    	roots = [sha3(x) for x in tree[level-1]]


    # Make a Merkle proof for some index
    def proof(self, index):













