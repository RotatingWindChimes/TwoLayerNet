import numpy as np


def cross_entropy_loss(y, t, delta=1e-7):
    # 需要将y和t都转换成二维形式(统一)
    if len(y.shape) == 1:
        y = y.reshape(1, y.size())
        t = t.reshape(1, t.size())
    batch_size = y.shape[0]

    return -np.sum(t * np.log(y+delta)) / batch_size


def softmax(x):
    x_max = np.max(x)

    x_exp = np.exp(x - x_max)
    x_exp_sum = np.sum(x_exp, axis=1, keepdims=True)

    return x_exp/x_exp_sum
