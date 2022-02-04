import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from dataset import ptb
from simple_rnnlm import SimpleRnnlm


if __name__ == "__main__":
    batch_size = 10
    wordvec_size = 100
    hidden_size = 100
    time_size = 5
    lr = 0.1
    max_epoch = 100

    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_size = 1000
    corpus = corpus[:corpus_size]
    vocab_size = int(max(corpus) + 1)
    xs = corpus[:-1]
    ts = corpus[1:]

    model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)
    trainer = RnnlmTrainer(model, optimizer)

    trainer.fit(xs, ts, max_epoch, batch_size, time_size)
    trainer.plot()
    plt.savefig("result_by_trainer")