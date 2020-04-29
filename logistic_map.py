import numpy as np
import matplotlib.pyplot as plt


def log_map_iterate(n, x, a):
    # function for logistic map
    for i in range(1, n):
        x = a * x * (1 - x)
    return x


def points():
    # generate points x and a
    sequence = np.arange(1, 4, .0001)
    a_list = [log_map_iterate(np.random.randint(100, 500), 0.6, a) for a in sequence]
    return sequence, a_list


def show_graph(a_list, x_list):
    # show scatter plot -  Visualize the bifurcation diagram
    plt.scatter(a_list, x_list, s=.05)
    plt.show()


def main():
    # RUN IT!
    r_list, a_list = points()
    show_graph(r_list, a_list)


if __name__ == '__main__':
    main()
