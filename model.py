import numpy as np
from collections import OrderedDict
from layers import Affine, SoftmaxWithLoss, Relu
import pickle


class TwoLayerNet:
    """ 两层神经网络 """
    def __init__(self, input_size, hidden_size, output_size, weight_decay_lambda, weight_init_std=0.01,
                 learning_rate_decay=0.001):
        # 参数
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["b2"] = np.zeros(output_size)

        # 层
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])

        self.softmax = SoftmaxWithLoss()

        # 正则化系数
        self.weight_decay_lambda = weight_decay_lambda
        # 学习率衰减率
        self.learning_rate_decay = learning_rate_decay
        # 隐藏层大小
        self.hidden_size = hidden_size

    # 预测函数这里我们只输出各个类的分数，没有进行归一化，这样可以节省时间
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)

        # 权重衰减
        weight_decay = 0
        for i in range(1, 3):
            W = self.params["W"+str(i)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.softmax.forward(y, t)

    # 只考虑正确个数
    def accuracy(self, x, t):
        y = np.argmax(self.predict(x), axis=1)
        if len(t.shape) == 2:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t)

        return accuracy

    def gradient(self, x, t):
        # 先前向传播一轮，存储中间值
        loss = self.loss(x, t)

        # 反向传播
        dout = self.softmax.backward(dout=1)

        # 反转层
        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        # 存储梯度
        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db

        return loss, grads

    def store(self, path="model_parameters.pkl"):
        with open(path, "wb") as fw:
            params = {"W1": self.params["W1"], "b1": self.params["b1"],
                      "W2": self.params["W2"], "b2": self.params["b2"]}

            pickle.dump(params, fw)

    def load(self, path="model_parameters.pkl"):
        with open(path, "rb") as fr:
            params = pickle.load(fr)

        self.params["W1"] = params["W1"]
        self.params["W2"] = params["W2"]
        self.params["b1"] = params["b1"]
        self.params["b2"] = params["b2"]
