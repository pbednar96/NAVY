import numpy as np
import matplotlib.pyplot as plt

LENGTH = 1000
ITER = 3

def fractal_landscape(length, r):
    # maybe a little weird approach

    # generate first line
    points = {}
    points[0] = np.random.randint(r - r / 2, r)
    points[length] = np.random.randint(r - r / 2, r)

    # generate middle points
    for iter in range(ITER):
        t = 2 ** iter
        for i in range(t):
            length_one_sequence = length / t
            point = length_one_sequence / 2 + i * length_one_sequence
            x = np.random.randint(r - r / 2, r)
            points[point] = x

    return points


def main():
    points = fractal_landscape(LENGTH, 200)
    x = sorted(points)
    y = [points[i] for i in x]

    points = fractal_landscape(LENGTH, 500)
    y1 = [points[i] for i in x]

    points = fractal_landscape(LENGTH, 1000)
    y2 = [points[i] for i in x]

    fig, ax = plt.subplots()
    ax.stackplot(x, y, y1, y2)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
