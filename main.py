from mnist import load_mnist
from model import TwoLayerNet
from train import train_model, test_model


def main():
    # 读入数据
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    # 可调超参数
    learning_rate = 0.1
    weight_decay = 1.0
    hidden_size = 100

    # 固定超参数
    input_size = 784
    output_size = 10
    n_epochs = 100
    batch_size = 100
    show_interval = 5

    # 设置模型
    network = TwoLayerNet(input_size=input_size, hidden_size=hidden_size,
                          output_size=output_size, weight_decay_lambda=weight_decay)

    # 模型训练
    train_model(model=network, batch_size=batch_size, n_epochs=n_epochs, learning_rate=learning_rate,
                show_interval=show_interval, train_data=x_train, train_label=t_train, test_data=x_test,
                test_label=t_test)

    # 模型保存
    network.store()

    # 加载模型并测试模型精度
    network.load()
    accuracy = test_model(model=network, batch_size=batch_size, data=x_test, label=t_test)
    print(f"The current accuracy of the two layer network is {accuracy}.")


if __name__ == '__main__':
    main()
