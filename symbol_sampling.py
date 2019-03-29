import numpy as np


# uniformly select s out of n symbols
def request_generation(n, s, replacement):
    if replacement == 0:
        return list(np.random.choice(n, s, replace=False))
    else:
        return list(np.random.choice(n, s))
