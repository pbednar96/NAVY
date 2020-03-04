import numpy as np
import matplotlib.pyplot as plt

sample = np.array([[0, 1, 1, 1, 0],
                   [0, 1, 0, 1, 0],
                   [0, 1, 1, 1, 0],
                   [0, 1, 0, 1, 0],
                   [0, 1, 1, 1, 0]
                   ])

to_recover = np.array([[0, 1, 1, 1, 1],
                       [1, 1, 0, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 0, 1, 0],
                       [1, 0, 1, 0, 0]
                       ])


# sample = np.array([[1, 0, 0, 0],
#                    [0, 1, 0, 0],
#                    [0, 0, 1, 0],
#                    [0, 0, 0, 1]
#                    ])

# to_recover = np.array([[1, 0, 1, 0],
#                        [0, 1, 0, 1],
#                        [0, 0, 0, 0],
#                        [0, 0, 1, 1]
#                        ])

# sample = np.array([[1, 0], [0, 1]])

# to_recover = np.array([[0, 1], [1, 0]])


def main():
    row, column = sample.shape
    sample_v = sample.flatten()
    sample_v[sample_v == 0] = -1
    sample_v = sample_v.reshape(1, len(sample_v))
    sample_v_T = sample_v.reshape(-1, 1)
    weighted_matrix = np.dot(sample_v_T, sample_v)

    np.fill_diagonal(weighted_matrix, 0)
    # print(weighted_matrix)

    # print("Sample:")
    # print(sample)
    # print("To recover:")
    # print(to_recover)
    to_recover_v = to_recover.flatten()
    to_recover_v[to_recover_v == 0] = -1
    to_recover_v = to_recover_v.reshape(1, len(to_recover_v))

    for i in range(row * column):
        value = np.dot(to_recover_v, weighted_matrix[:, i])
        tmp = np.sign(value)

        to_recover_v[0][i] = tmp

    to_recover_np_array = np.asarray(to_recover_v)
    to_recover_np_array = np.reshape(to_recover_np_array, (row, column))
    to_recover_np_array[to_recover_np_array == -1] = 0

    # print("Result:")
    # print(to_recover_np_array)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.matshow(sample)
    ax1.set_title("Sample")
    ax2.matshow(to_recover)
    ax2.set_title("To recover")
    ax3.matshow(to_recover_np_array)
    ax3.set_title("After recover")
    plt.show()


if __name__ == "__main__":
    main()
