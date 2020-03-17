import gym
import time
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

LEARNING_RATE = .001
SIZE_DATASET = 3000

# select env
env = gym.make("CartPole-v0")
env.reset()


def create_dateset():
    # create dataset for training with random actions, nevertheless will be add only actions
    # with reward higher then 40
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(SIZE_DATASET):
        env.reset()
        score = 0
        game_memory = []
        prev_observation = []

        # restriction max iteration random action < 200
        for _ in range(200):
            # random action
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                break
        # score/sum_rewards is higher then 40
        if score >= 40:
            accepted_scores.append(score)
            for data in game_memory:
                # output for NN
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]
                training_data.append([data[0], output])

        scores.append(score)

    return training_data


def create_model():
    # create NN model with Keras NN
    # ARCH -> 4-24-24-2
    # len(env.observation_space) = 4
    # env.action_space.n = 2
    model = Sequential()
    model.add(Dense(24, input_dim=4, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(2, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
    return model


def train_model(training_data):
    # train process

    # list with observation data
    observation_state = [i[0] for i in training_data]
    # list with output data
    target = [i[1] for i in training_data]

    model = create_model()
    # train 5 epochs
    for _ in range(5):
        # for each item in list
        for state, target_f in zip(observation_state, target):
            state = np.array(state).reshape(1, 4)
            target_f = np.array(target_f).reshape(1, 2)
            model.fit(state, target_f, epochs=1, verbose=0)
    return model


def main():
    train_data = create_dateset()
    print("Dataset was created!")
    model = train_model(train_data)
    print("Training done!")
    # 10 iteration with train model
    for i in range(10):
        env.reset()
        env.render()
        score = 0
        prev_observation = []
        for _ in range(200):
            env.render()
            time.sleep(0.03)

            #first action = random action
            if len(prev_observation) == 0:
                action = env.action_space.sample()
            else:
                # action with higher predicted value
                action = np.argmax(model.predict(prev_observation.reshape(1, 4))[0])
            # make chose action
            new_observation, reward, done, info = env.step(action)
            prev_observation = new_observation
            # print(prev_observation.reshape(1, 4))

            score += reward
            if done:
                break

        print(f"{i}: score:{score}")


if __name__ == "__main__":
    print(env.action_space.n)
    main()

# RESULT
# Training done!
# 0: score:102.0
# 1: score:200.0
# 2: score:185.0
# 3: score:139.0
# 4: score:200.0
# 5: score:125.0
# 6: score:200.0
# 7: score:121.0
# 8: score:200.0
# 9: score:129.0
