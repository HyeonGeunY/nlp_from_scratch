import numpy as np

from common.functions import softmax, cross_entropy_error


class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        (W,) = self.params  # 언패킹 W, : 중요
        self.x = x
        return self.x @ W

    def backward(self, dout):
        (W,) = self.params
        dx = dout @ W.T
        dw = self.x.T @ dout
        self.grads[0][...] = dw  # [...] : 메모리 위치 고정, 깊은 복사
        return dx  # 연쇄 법칙으로 넘겨주기 위한 값


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 - np.exp(x))

        return self.out

    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx
    
class SigmoidWithLoss: # ??
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None
        self.x = None
        
    def forward(self, x, y):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))
        
        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)
        
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        
        dx = (self.y - self.t) * dout / batch_size
        return dx


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = x @ W + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params

        dW = x.T @ dout
        dx = dout @ W.T
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= sumdx * self.out
        return dx
    
class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None # softmax 의 출력
        self.t = None # 정답
    
    def forward(self, x, y):
        self.t = t
        self.y = softmax(x)
        
        # 정답레이블이 원 핫일 경우 정답 인덱스로 변환
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)
            
        loss = cross_entropy_error(self.y, self.t)
        
        return loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1 # y-t 만들기
        dx *= dout
        dx = dx / batch_size
        
        return dx
    
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        
    def forward(self, x, train_fig=True):
        if train_fig:
            self.mask = np.random.random(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
        
    def backward(self, dout):
        return dout * self.mask
