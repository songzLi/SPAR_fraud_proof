import pickle
import time
from os import listdir
from os.path import isfile, join


def convert_file_to_parity(file_name):
    start = time.time()
    raw = open(file_name, 'r').read()
    parties_lines = raw.splitlines()
    parities = [[]] * len(parties_lines)
    for i in range(len(parties_lines)):
        # if i % 100 == 0:
            # print(i, 'out of', str(len(parties_lines)))
        line = parties_lines[i].split()
        parities[i] = [int(idx) for idx in line]
    print('Converted. Took ' + str(time.time() - start) + 'seconds')
    start = time.time()
    # extract the length (K) of the LDPC code from its file name
    name = file_name.split('.')[0]
    integers = [str(i) for i in range(10)]
    length = ''.join([n for n in name if n in integers])
    print(length)
    with open('parities_' + length + '.pickle',
              'wb') as handle:
        pickle.dump(parities, handle)
    print('Converted ' + file_name + ' to parity matrix and stored. Took ' +
          str(time.time() - start) + 'seconds.')


def convert_all_txt_files():
    files = [f for f in listdir('./')
             if isfile(join('./', f)) and f[-4:] == '.txt']
    for file in files:
        convert_file_to_parity(file)


def convert_parity_to_symbol_core(parities, R=0.25):
    P = len(parities)
    print('num parities', P)
    N = int(P / (1 - R))
    symbols = [[] for i in range(N)]
    for i in range(len(parities)):
        p = parities[i]
        # print('parity', i, 'consists of', p)
        for idx in p:
            symbols[idx].append(i)
    print(symbols[0])
    return symbols


def convert_all_parity_files_to_symbols():
    files = [f for f in listdir('./')
             if isfile(join('./', f)) and f[-7:] == '.pickle' and
             f[:8] == 'parities']
    for file in files:
        with open(file, 'rb') as handle:
            parities = pickle.load(handle)
        start = time.time()
        symbols = convert_parity_to_symbol_core(parities)
        print('converted', file, 'in',
              str(time.time() - start), 'seconds')
        result = {}
        result['parities'] = parities
        result['symbols'] = symbols
        with open('symbols_and_parities_' + file[9:-7] +
                  '.pickle', 'wb') as handle:
            pickle.dump(result, handle)


convert_all_txt_files()
convert_all_parity_files_to_symbols()
