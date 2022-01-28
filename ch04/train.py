import sys
sys.path.append('..')
import numpy as np
from common import config

import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from cbow import CBOW
from skip_gram import SkipGram
from common.util import create_contexts_target, to_cpu, to_gpu
from dataset import ptb

WINDOW_SIZE = 5
HIDDEN_SIZE = 100
BATCH_SIZE = 100
MAX_EPOCH = 10

if __name__ == "__main__":
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    vocab_size = len(word_to_id)

    contexts, target = create_contexts_target(corpus, WINDOW_SIZE)
    
    if config.GPU:
        contexts, target = to_gpu(contexts), to_gpu(target)

    #model = CBOW(vocab_size, HIDDEN_SIZE, WINDOW_SIZE, corpus)
    model = SkipGram(vocab_size, HIDDEN_SIZE, WINDOW_SIZE, corpus)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)
    
    trainer.fit(contexts, target, MAX_EPOCH, BATCH_SIZE)
    trainer.plot()

    word_vecs = model.word_vecs
    if config.GPU:
        word_vecs = to_cpu(word_vecs)
    params = {}
    params['word_vecs'] = word_vecs.astype(np.float16)
    params['word_to_id'] = word_to_id
    params['id_to_word'] = id_to_word
    pkl_file = 'cbow_params.pkl'

    with open(pkl_file, 'wb') as f:
        pickle.dump(params, f, -1)    
        
        

    
