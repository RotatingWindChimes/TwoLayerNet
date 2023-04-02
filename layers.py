import numpy as np
from functions import cross_entropy_loss, softmax


class MulLayer:
    """ 乘法层 """
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y

        return self.x * self.y

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    """ 加法层 """
    def __init(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy


class Relu:
    """ relu层 """
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0

        return dout


class Sigmoid:
    """ sigmoid层 """
    def __init(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)

        return dx


class Affine:
    """ affine层 """
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class SoftmaxWithLoss:
    """ softmax层 """
    def __init__(self):
        self.loss = None   # 损失
        self.y = None      # softmax函数输出
        self.t = None      # target

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_loss(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]    # 批量大小
        dx = (self.y - self.t) / batch_size

        return dx
