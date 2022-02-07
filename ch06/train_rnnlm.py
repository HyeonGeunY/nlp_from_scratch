import sys
sys.path.append('..')
from common.optimizer import SGD
from common.trainer import RNNlmTrainer
from common.util import eval_perplexity
from dataset import ptb
from rnnlm import Rnnlm


if __name__ == "__main__":
    batch_size = 20
    wordvec_size = 100
    hidden_size = 100 # RNN의 은닉 상태 백터의 원소 수
    time_size = 35    # RNN을 펼치는 크기
    lr = 20.0
    max_epoch = 4
    max_grad = 0.25

    # 학습 데이터 읽기
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_test, _, _ = ptb.load_data('test')
    vocab_size = len(word_to_id)
    xs = corpus[:-1]
    ts = corpus[1:]

    model = Rnnlm(vocab_size=vocab_size, wordvec_size=wordvec_size, hidden_size=hidden_size)
    optimizer = SGD(lr)
    trainer = RNNlmTrainer(model, optimizer)

    trainer.fit(xs, ts, max_epoch=max_epoch, batch_size=batch_size, time_size=time_size, max_grad=max_grad, eval_interval=20)
    trainer.plot(ylim=(0, 500))

    model.reset_state()
    ppl_test = eval_perplexity(model, corpus_test)
    print(f'테스트 퍼플렉서티: {ppl_test}')

    model.save_params()



