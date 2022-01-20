import sys
from typing import Optional
sys.path.append('..')
import numpy as np
import time
import matplotlib.pyplot as plt

from common.np import *
from common.util import clip_grads

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()

        for epoch in range(max_epoch):
            idx = np.random.permutation(np.arange(data_size)) # 매 epochs 마다 랜덤하게 데이터를 섞는다.
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]

                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print(f'| epcoh {self.current_epoch + 1}, iters {iters + 1}/{max_iters} | time {elapsed_time} | loss {avg_loss}')
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel(f'iters (x {self.eval_interval})')
        plt.ylabel('loss')
        plt.show()

def remove_duplicate(params, grads):
    """
    매개변수 배열 중 중복되는 가중치를 하나로 모아 그 가중치에 대응하는 기울기를 더한다.
    """
    params, grads = params[:], grads[:]

    while True:
        find_fig = False
        L = len(params)

        for i in range(0, L-1):
            for j in range(i + 1, L):
                if params[i] is params[j]:
                    grads[i] += grads[j]
                    find_fig = True
                    params.pop(j)
                    grads.pop(j)

                # 가중치를 전치행렬로 공유하는 case
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_fig = True
                    params.pop(j)
                    grads.pop(j)

                if find_fig: break
            if find_fig: break
        if not find_fig: break

    return params, grads



