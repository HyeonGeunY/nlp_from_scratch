import sys
sys.path.append('..')
from common.np import *
from common.layers import Embedding, SigmoidWithLoss
import collections

class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W) # W 는 embedding을 사용하지 않을 때의 W의 transpose 형태
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        '''
        Parameters
        ------------
        h : np.ndarray
            은닉 뉴런
        '''
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)

        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h  # 곱의 역전파 (상대방을 곱한다.)
        self.embed.backward(dtarget_W)
        dh = dout * target_W # 곱의 역전파 (상대방을 곱한다.)
        return dh

class UnigramSampler:
    def __init_(self, corpus, power, samle_size):
        