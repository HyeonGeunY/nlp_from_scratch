from ssl import HAS_TLSv1
from common.np import *
from common.layers import *
from common.functions import sigmoid


class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next ** 2)
        db = np.sum(dt, axis=0)
        dWh = np.dot(h_prev.T, dt)
        dh_prev = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev


class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.dh = None, None
        self.stateful = stateful

    def forward(self, xs):
        """
        Parameters
        -----------
        xs :
            Time T개 분량의 input (N, T, D), N : batch_size, T : T개 분량의 시계열 데이터, D : 입력 벡터의 차원 수
        """
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype="f")

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype="f")

        for t in range(T):
            layer = RNN(*self.params)  # 실제로는 하나의 가중치를 사용하는 층 하나 (나중에 역전파시 모든 grads를 합 한다.)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs  # hs?

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype="f")
        dh = 0
        grads = [0, 0, 0]

        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)  # 분기 노드의 역전파 (합)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad  # 실제로는 같은 가중치를 갖는 RNN층을 상태 t에 대해서 반복 -> ~t 상태까지의 grads 누적한다.
        self.dh = dh

        return dxs

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None


class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        N, T = xs.shape
        V, D = self.W.shape

        out = np.empty((N, T, D), dtype="f")
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]
        self.grads[0][...] = grad
        return None


class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape
        W, b = self.params

        rx = x.reshape(N * T, -1)
        out = np.dot(rx, W) + b
        self.x = x
        return out.reshape(N, T, -1)

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N * T, -1)
        rx = x.reshape(N * T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 정답 레이블이 원 핫 벡터인 경우
            ts = ts.argmax(axis=2)

        mask = ts != self.ignore_label

        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # ignore_label에 해당하는 데이터는 손실을 0으로 설정
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # ignore_label에 해당하는 데이터 => 기울기 0으로 설정

        dx = dx.reshape((N, T, V))

        return dx


class LSTM:
    def __init__(self, Wx, Wh, b):
        """
        Parameters:
        Wx : 입력에 대한 가중치 (4개 분의 가중치가 담겨 있음)
        Wh : 은닉 상태 h에 대한 가중치 (4개 분의 가중치)
        b : 편향 (4개분의 가중치)
        """

        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape
        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        f = A[:, :H]
        g = A[:, H : 2 * H]
        i = A[:, 2 * H : 3 * H]
        o = A[:, 3 * H :]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tanh_c_next = np.tanh(c_next)

        ds = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)

        dc_prev = ds * f

        di = ds * g
        df = ds * c_prev
        do = dh_next * tanh_c_next
        dg = ds * i

        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= 1 - g ** 2

        dA = np.hstack((df, dg, di, do))

        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev


class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N, T, H), dtype="f")

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype="f")
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype="f")

        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h

            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype="f")
        dh, dc = 0, 0

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None


class TimeDropout:
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_fig = True

    def forward(self, xs):
        if self.train_fig:
            flg = np.random.rand(*xs.shape) > self.dropout_ratio
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = flg.astype(np.float32) * scale
            return xs * self.mask
        else:
            return xs

    def backward(self, dout):
        return dout * self.mask

    class GRU:
        def __init__(self, Wx, Wh):
            self.Wx, self.Wh = Wx, Wh
            self.dWx, self.dWh = None, None
            self.cache = None

        def forward(self, x, h_prev):
            H, H3 = self.Wh.shape  # 기본 Wh dim : (H * H) => 빠른 계산을 위해 (H * 3H)
            Wxz, Wxr, Wx = self.Wx[:, :H], self.Wx[:, H : 2 * H], self.Wx[:, 2 * H :]
            Whz, Whr, Wh = self.Wh[:, :H], self.Wh[:, H : 2 * H], self.Wh[:, 2 * H :]

            z = sigmoid(np.dot(x, Wxz) + np.dot(h_prev, Whz))
            r = sigmoid(np.dot(x, Wxr) + np.dot(h_prev, Whr))
            h_hat = np.tanh(np.dot(x, Wx) + np.dot(r * h_prev, Wh))
            h_next = (1 - z) * h_prev + z * h_hat

            self.cache = (x, h_prev, z, r, h_hat)

            return h_next

        def backward(self, dh_next):
            H, H3 = self.Wh.shape
            Wxz, Wxr, Wx = self.Wx[:, :H], self.Wx[:, H : 2 * H], self.Wx[:, 2 * H :]
            Whz, Whr, Wh = self.Wh[:, :H], self.Wh[:, H : 2 * H], self.Wh[:, 2 * H :]
            x, h_prev, z, r, h_hat = self.cache

            dh_hat = dh_next * z
            dh_prev = dh_next * (1 - z)

            # tanh
            dt = dh_hat * (1 - h_hat ** 2)
            dWh = np.dot((r * h_prev).T, dt)
            dhr = np.dot(dt, Wh.T)
            dWx = np.dot(x.T, dt)
            dx = np.dot(dt, Wx.T)
            dh_prev += r * dhr

            # update gate(z)
            dz = dh_next * h_hat - dh_next * h_prev
            dt = dz * z * (1 - z)
            dWhz = np.dot(h_prev.T, dt)
            dh_prev += np.dot(dt, Whz.T)
            dWxz = np.dot(x.T, dt)
            dx += np.dot(dt, Wxz.T)

            # reset gate
            dr = dhr * h_prev
            dt = dr * r * (1 - r)
            dWhr = np.dot(h_prev.T, dt)
            h_prev += np.dot(dt, Whr.T)
            dWxr = np.dot(x.T, dt)
            dx += np.dot(dt, Wxr.t)

            self.dWx = np.hstack((dWxz, dWxr, dWx))
            self.dWh = np.hstack((dWhz, dWhr, dWh))

            return dx, dh_prev
