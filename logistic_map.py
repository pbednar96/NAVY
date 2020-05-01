import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model

TRAIN = False


def create_model():
    # create NN model with Keras NN
    # ARCH -> 2-12-24-12-1
    # input : a, x
    # output : y
    # activation func tanh and sigmoid have a better shapes for this task
    model = Sequential()
    model.add(Dense(12, input_dim=2, activation='tanh'))
    model.add(Dense(24, activation='tanh'))
    model.add(Dense(12, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer=Adam(lr=0.0001))
    return model


def train_model(inputs, targets):
    # train process
    model = create_model()
    # train epochs
    # I'm not sure, how improve performance, usually throw error
    for e in range(40):
        print(e)
        for input_NN, target in zip(inputs, targets):
            input_NN = np.reshape(input_NN, (1, 2))
            target = np.reshape(target, (1,))
            model.fit(input_NN, target, epochs=1, verbose=0)

    return model


def log_map_iterate(x, a):
    # function for logistic map
    n = np.random.randint(100, 500)
    for i in range(1, n):
        x = a * x * (1 - x)
    return x


def points():
    # generate points x and a
    sequence = np.arange(1, 4, .001)
    x_list = [log_map_iterate(0.6, a) for a in sequence]
    return sequence, x_list


def show_graph(a_list, x_list):
    # show scatter plot -  Visualize the bifurcation diagram
    plt.scatter(a_list, x_list, s=.05)
    plt.axis(xlim=(0, 1), ylim=(1, 4))
    plt.show()


def main():
    # RUN IT!
    a_list, x_list = points()
    show_graph(a_list, x_list)


    x = 0.6
    data_input = [[x, a] for a in a_list]
    print(data_input)
    if TRAIN:
        model = train_model(data_input, x_list)
        model.save('my_model_X.h5')
    else:
        model = load_model('my_model.h5')

    sequence_NN = np.arange(1, 4, .0001)
    x_list_NN = []
    for i in sequence_NN:
        a = np.array([0.6, i])
        a = a.reshape(1, 2)
        x_list_NN.append(model.predict(a))
    show_graph(sequence_NN, x_list_NN)


if __name__ == '__main__':
    main()
