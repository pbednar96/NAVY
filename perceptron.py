import random
import matplotlib.pyplot as plt
import numpy as np

LEARNING_RATE = 0.01
STEPS = 10000


def train_generate_point():
    data = []
    for i in range(100):
        x1 = random.uniform(-10, 10)
        x2 = random.uniform(-35, 35)
        # y = 4x - 5
        if x2 >= 4 * x1 - 5:
            y = 1
        else:
            y = -1
        data.append([1.0, x1, x2, y])
    return data


def test_generate_point():
    data = []
    for i in range(1000):
        x1 = random.uniform(-10, 10)
        x2 = random.uniform(-35, 35)
        data.append([1.0, x1, x2])
    return data


def show_scatter(data):
    x1 = [item[1] for item in data if item[3] == 1]
    x2 = [item[1] for item in data if item[3] == -1]
    y1 = [item[2] for item in data if item[3] == 1]
    y2 = [item[2] for item in data if item[3] == -1]
    plt.scatter(x1, y1, color="red")
    plt.scatter(x2, y2, color="green")
    line = np.linspace(-10, 10)
    plt.plot(line, line * 4 - 5, color="black")
    plt.show()


def sigma(value):
    if value > 0:
        return 1
    elif value == 0:
        return 0
    else:
        return -1


def perceptron(train_data, w):
    for i in train_data:
        suma_weight = w[0] * i[0] + w[1] * i[1] + w[2] * i[2]
        suma_weight = np.sign(suma_weight)
        if suma_weight <= 0:
            error = i[3] - suma_weight
            for x in range(len(w)):
                w[x] = w[x] + error * i[x] * LEARNING_RATE
    return w


def testing(w, test_data):
    new_data = []
    for i in test_data:
        suma_weight = np.dot(w, i)
        suma_weight = np.sign(suma_weight)
        new_data.append([i[0], i[1], i[2], suma_weight])
    return new_data


def main():
    train_data = train_generate_point()
    test_data = test_generate_point()
    w = [random.random() for x in range(3)]
    for _ in range(STEPS):
        w = perceptron(train_data, w)
    test_data = testing(w, test_data)
    show_scatter(test_data)


if __name__ == "__main__":
    main()
