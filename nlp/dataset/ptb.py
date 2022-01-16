import sys
import os
sys.path.append('..')
try:
    import urllib.request
except ImportError:
    raise ImportError('Use python3!')

import pickle
import numpy as np

url_base = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/'
key_file = {
    'train': 'ptb.train.txt',
    'test': 'ptb.test.txt',
    'valid': 'ptb.valid.txt'
}

save_file = {
    'train': 'ptb.train.npy',
    'test': 'ptb.test.npy',
    'valid': 'ptb.valid.npy'
}

vocab_file = 'ptb.vocab.pkl'

dataset_dir = os.path.dirname(os.path.abspath(__file__))

def _download(file_name):
    file_path = dataset_dir + '/' + file_name
    if os.path.exists(file_name):
        return

    print()