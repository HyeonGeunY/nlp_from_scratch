import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)

def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True) # softmax는 인풋인자에 특정 수를 더하거나 곱하는 것에 영향을 받지 않는다.
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
        
    if x.ndim == 1:
        x = x - x.max()
        x = np.exp(x)
        x /= x.sum()
        
    return x

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    # if 정답 데이터가 원핫 => 레이블 인덱스로 변경
    if t.size == y.size:
        t = t.argmax(axis=1)
    
    batch_size = y.shape[0]
    
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
        
        
    
