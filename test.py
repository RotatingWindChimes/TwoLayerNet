from train import data_iter
from model import TwoLayerNet
from mnist import load_mnist


def test_model():

    # 获得数据
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    batch_size = 100

    # 实例化模型
    network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10, weight_decay_lambda=2.0)

    # 加载模型并测试模型精度
    network.load()

    test_correct = 0

    for X, t in data_iter(batch_size, x_test, t_test):
        test_correct += network.accuracy(X, t)

    print(f"The selected model has an accuracy of {float(test_correct) / len(t_test)}")


test_model()
