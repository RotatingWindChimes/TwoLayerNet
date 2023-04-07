import numpy as np
import matplotlib.pyplot as plt


def learning_rate_decay(initial_rate, cur_time, method, beta):
    """ 学习率衰减 """
    if method == "Inverse Decay":
        return initial_rate / (1.0 + beta * cur_time)
    elif method == "Exponential Decay":
        return initial_rate * np.exp(np.log(beta) * cur_time)
    elif method == "Natural Exponentail Decay":
        return initial_rate * np.exp(-beta * cur_time)
    else:
        print("Not valid method.")


def data_iter(batch_size, features, labels):
    # 数据集大小
    datasize = len(labels)

    # 随机数种子，方便复现
    np.random.seed(42)
    indices = np.random.permutation(range(datasize))

    for i in range(0, datasize, batch_size):
        batch_indices = indices[i: min(i+batch_size, datasize)]

        yield features[batch_indices], labels[batch_indices]


def sgd(model, lr, grads):
    for key in ["W1", "W2", "b1", "b2"]:
        if key[0] == "W":
            model.params[key] -= lr * (grads[key] + model.weight_decay_lambda * model.params[key])
        else:
            model.params[key] -= lr * grads[key]


def test_model(model, batch_size, data, label):
    test_correct = 0

    for X, t in data_iter(batch_size, data, label):
        test_correct += model.accuracy(X, t)

    return float(test_correct) / len(label)


def train_model(model, batch_size, n_epochs, learning_rate, show_interval, train_data, train_label, test_data,
                test_label):

    loss_dict = {"train": [], "test": []}
    test_acc = []

    for epoch in range(n_epochs):
        # 当前模型精度
        test_accuracy = test_model(model, batch_size, test_data, test_label)
        test_acc.append(test_accuracy)

        # 当前模型的训练损失、测试损失
        for name, data, label in [("train", train_data, train_label), ("test", test_data, test_label)]:
            total_loss = 0.0
            iter_time = 0
            for X, t in data_iter(batch_size, data, label):
                loss, grads = model.gradient(X, t)

                # 总损失更新
                total_loss += loss
                iter_time += 1

                if name == "train":
                    # 学习率更新
                    lr = learning_rate_decay(learning_rate, epoch, method="Inverse Decay",
                                             beta=model.learning_rate_decay)

                    # 参数更新
                    sgd(model, lr, grads)

            # 平均损失更新
            loss_dict[name].append(total_loss / iter_time)

        # 当前模型的测试精度
        if epoch % show_interval == 0:
            print(f"Epoch {epoch}".center(60, "*"))
            print("Train Loss: {}".format(loss_dict["train"][-1]), "Test Loss: {}".format(loss_dict["test"][-1]))
            print(f"Test Accuracy: {test_acc[-1]}")

    plt.plot(range(n_epochs), loss_dict["train"], linestyle="-.", c="cornflowerblue", label="Train Loss")
    plt.plot(range(n_epochs), loss_dict["test"], linestyle="--", c="turquoise", label="Test Loss")
    plt.plot(range(n_epochs), test_acc, c="k", label="Test Accuracy")

    plt.xticks(range(0, n_epochs, 5))
    plt.xlabel("Iteration Times")
    plt.ylabel("Loss", rotation=0)
    plt.title(f"LR: {learning_rate}, Coef: {model.weight_decay_lambda}, Hidden: {model.hidden_size}")

    plt.legend()

    plt.savefig("result.jpg")
    plt.show()
