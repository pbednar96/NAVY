import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

A_I = np.array([[0.00, 0.00, 0.01, 0.00, 0.26, 0.00, 0.00, 0.00, 0.05],
                [0.20, -0.26, -0.01, 0.23, 0.22, -0.07, 0.07, 0.00, 0.24],
                [-0.25, 0.28, 0.01, 0.26, 0.24, -0.07, 0.07, 0.00, 0.24],
                [0.85, 0.04, -0.01, -0.04, 0.85, 0.09, 0.00, 0.08, 0.84]
                ])

J_L = np.array([[0, 0, 0], [0, 0.8, 0], [0, 0.22, 0], [0, 0.8, 0]])

P = np.array([.05, .25, 0.35, 0.35])


def main():
    x = np.array([0, 0, 0])
    f1 = A_I[0].reshape(3, 3)
    f2 = A_I[1].reshape(3, 3)
    f3 = A_I[2].reshape(3, 3)
    f4 = A_I[3].reshape(3, 3)
    A = np.array(x[0])
    B = np.array(x[1])
    C = np.array(x[2])

    for _ in range(20000):
        A = np.append(A, x[0])
        B = np.append(B, x[1])
        C = np.append(C, x[2])
        x.reshape(-1, 1)

        rand_num = np.random.random()
        if rand_num < P[0]:
            x = np.dot(f1, x) + J_L[0]
        elif rand_num < (P[0] + P[1]):
            x = np.dot(f2, x) + J_L[1]
        elif rand_num < (P[0] + P[1] + P[2]):
            x = np.dot(f3, x) + J_L[2]
        else:
            x = np.dot(f4, x) + J_L[3]

        x.reshape(1, 3)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    print("done")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.scatter3D(A.flatten(), B.flatten(), C.flatten(), c='k', s=1)
    plt.show()


if __name__ == "__main__":
    main()
