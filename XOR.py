import numpy as np
import random

LEARNING_RATE = 0.1
STEPS = 10000


def sigmoid(input_x):
    return 1 / (1 + np.exp(-input_x))


def sigmoid_(input_x):
    return input_x * (1 - input_x)


def test(wh, wo, item):
    hidden_activation = np.dot(item, wh.T)
    hidden_output = sigmoid(hidden_activation)
    hidden_output = np.concatenate((np.ones(1), hidden_output))

    output_activation = np.dot(hidden_output, wo)
    predicted_output = sigmoid(output_activation)
    return predicted_output


def main():
    # XOR data bias,x1,x2
    x = np.array([[1, 0, 0],
                  [1, 0, 1],
                  [1, 1, 0],
                  [1, 1, 1]])

    y = np.array([[0], [1], [1], [0]])
    wh = np.random.randn(2, 3)
    wo = np.random.randn(3, 1)
    for _ in range(STEPS):
        for index, item in enumerate(x):
            hidden_activation = np.dot(item, wh.T)
            hidden_output = sigmoid(hidden_activation)
            hidden_output = np.concatenate((np.ones(1), hidden_output))

            output_activation = np.dot(hidden_output, wo)
            predicted_output = sigmoid(output_activation)

            # error = 1/2 * (y[index] - predicted_output) ** 2
            error = y[index] - predicted_output
            delta = error * sigmoid_(predicted_output)
            # print(delta)

            delta_hidden1 = sigmoid_(hidden_output[1]) * delta * wo[1]
            delta_hidden2 = sigmoid_(hidden_output[2]) * delta * wo[2]
            # print(item)

            # print(wh)
            wo[1][0] += delta * hidden_output[1] * LEARNING_RATE
            wo[2][0] += delta * hidden_output[2] * LEARNING_RATE
            wo[0][0] += delta * LEARNING_RATE

            wh[0][1] += delta_hidden1 * item[1] * LEARNING_RATE
            wh[0][2] += delta_hidden1 * item[1] * LEARNING_RATE
            wh[0][0] += delta_hidden1 * LEARNING_RATE

            wh[1][1] += delta_hidden2 * item[1] * LEARNING_RATE
            wh[1][2] += delta_hidden2 * item[1] * LEARNING_RATE
            wh[1][0] += delta_hidden2 * LEARNING_RATE

    test_data = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    for i in test_data:
        x = test(wh, wo, i)
        print(f"{i[1:]}:{x}")


if __name__ == "__main__":
        np.random.seed(24)
        main()
