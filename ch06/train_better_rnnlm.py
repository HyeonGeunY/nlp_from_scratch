import sys
sys.path.append('..')
from common import config
from common.optimizer import SGD
from common.trainer import RNNlmTrainer
from common.util import eval_perplexity, to_gpu
from dataset import ptb
from train_better_rnnlm import BetterRnnlm

if __name__ == "__main__":
    batch_size = 20
    wordvec_size = 650
    hidden_size = 650
    time_size = 35
    lr = 20.0
    max_epoch = 40
    max_grad = 0.25
    dropout = 0.5
    
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_val, _, _ = ptb.load_data('val')
    corpus_test, _, _ = ptb.load_data('test')
    
    if config.GPU:
        corpus = to_gpu(corpus)
        corpus_val = to_gpu(corpus_val)
        corpus_test = to_gpu(corpus_test)
    
    vocab_size = len(word_to_id)
    xs = corpus[:-1]
    ts = corpus[1:]
        
    model = BetterRnnlm(vocab_size, wordvec_size, hidden_size, dropout)
    
    