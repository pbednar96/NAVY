import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy

EPISODE = 50
LEARNING_RATE = 0.01
GAMMA = 0.9

env_matrix = np.array([[0, 0, 0, 0, 0],
                       [0, -1, 0, 0, -1],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 10, 0],
                       [0, 0, 0, 0, 0]
                       ])


# ENV action

def update_state(state, direction):
    if direction == 0:
        state[0] -= 1
    elif direction == 1:
        state[0] += 1
    elif direction == 2:
        state[1] -= 1
    elif direction == 3:
        state[1] += 1
    if state == [1, 1] or state == [1, 4]:
        done = True

    elif state == [3, 3]:
        done = True
    else:
        done = False
    reward = env_matrix[state[0]][state[1]]
    return state, reward, done


def env_restriction(q_table):
    for x in range(5):
        for y in range(5):
            if y <= 0:
                q_table_item = y * 5 + x
                q_table[q_table_item][0] = -100
            if y >= 4:
                q_table_item = y * 5 + x
                q_table[q_table_item][1] = -100
            if x <= 0:
                q_table_item = y * 5 + x
                q_table[q_table_item][2] = -100
            if x >= 4:
                q_table_item = y * 5 + x
                q_table[q_table_item][3] = -100

    return np.array(q_table)


def init_random_place_in_env():
    init_state_x = np.random.randint(5)
    allow_state_list = np.where(env_matrix[init_state_x] == 0)
    init_state_y = np.random.choice(allow_state_list[0], 1)
    return [init_state_x, init_state_y[0]]


def show_env(list_all_state):
    fig, ax = plt.subplots()
    points = list_all_state[0]

    def animate(state):
        tmp_matrix = copy.deepcopy(env_matrix)
        tmp_matrix[state[0]][state[1]] = 10
        fig, ax = plt.subplots()
        matrice = ax.matshow(tmp_matrix)
        plt.colorbar(matrice)

    ani = animation.FuncAnimation(fig, animate, frames=len(points), interval=400, repeat=False)
    plt.show()


def main():
    ROWS, COLUMN = env_matrix.shape
    q_table = np.zeros((ROWS * COLUMN, 4))
    q_table = env_restriction(q_table)

    # print(q_table)
    # TRAIN and set Q table
    max_step = 20
    for _ in range(EPISODE):
        state = init_random_place_in_env()
        # print(q_table)
        max_step = 20
        while True:
            state_item_in_q = state[0] * 5 + state[1]
            all_direction = np.where(q_table[state_item_in_q] >= 0)
            direction = np.random.choice(all_direction[0], 1)
            #
            # print(state)
            # print(q_table[state_item_in_q])
            # print(all_direction)
            # print(direction)
            new_state, reward, done = update_state(state, direction[0])
            new_state_item_in_q = new_state[0] * 5 + new_state[1]
            q_table[state_item_in_q][direction[0]] += LEARNING_RATE * (
                    reward + GAMMA * max(q_table[new_state_item_in_q]))
            state = new_state

            if done == True:
                break
            if max_step == 0:
                break

    # print(q_table)

    # TEST
    all_state = []
    for i in range(10):
        state = init_random_place_in_env()
        max_iter = 15
        list_state = []
        while True:
            max_iter -= 1
            print(state)
            list_state.append(copy.deepcopy(state))

            tmp_matrix = copy.deepcopy(env_matrix)
            tmp_matrix[state[0]][state[1]] = 6
            plt.imshow(np.reshape(tmp_matrix, (5, 5)))
            plt.pause(0.2)

            state_item_in_q = state[0] * 5 + state[1]
            # print(q_table[state_item_in_q])
            direction = np.argmax(q_table[state_item_in_q])
            # print(direction)
            new_state, reward, done = update_state(state, direction)

            if new_state == [3, 3]:
                print("aim")
                break
            if new_state == [1, 1] or state == [1, 4]:
                print("death")
                break
            if max_iter == 0:
                break
            state = new_state
        all_state.append(list_state)


if __name__ == "__main__":
    main()
