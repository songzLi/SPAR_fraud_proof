import copy
import poly_utils
import rlp


# Every point is an element of GF(2**16), so represents two bytes
POINT_SIZE = 2
# Every symbol contains 128 points
POINTS_IN_SYMBOL = 128
# A symbol is 256 bytes
SYMBOL_SIZE = POINT_SIZE * POINTS_IN_SYMBOL



