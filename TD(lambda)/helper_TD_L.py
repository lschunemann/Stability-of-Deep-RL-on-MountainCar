import numpy as np
from TileCoding import *


def greedy_choice(w, state, numtilings, actions, hashtable, x_bound, v_bound):
    action_values = []
    for action in actions:
        action_values.append(value(state, w, numtilings, action, hashtable, x_bound, v_bound))
    return actions[np.argmax(action_values)]


# def cost_to_go(state, actions, w, numtilings, hashtable, x_bound, v_bound):
#     costs = []
#     for action in actions:
#         costs.append(value(state, numtilings, w, action, hashtable, x_bound, v_bound))
#     return -np.max(costs)


def epsilon_greedy(env, eps, w, numtilings, actions, state, hashtable, x_bound, v_bound):
    if np.random.random() < eps:
        return env.action_space.sample()
    action_values = []
    for action in actions:
        action_values.append(value(state, w, numtilings, action, hashtable, x_bound, v_bound))
    return actions[np.argmax(action_values)]


def value(state, w, numtilings, action, hashtable, x_bound, v_bound):
    tiles = get_active_tiles(hashtable, numtilings, state, action, x_bound, v_bound)
    return np.sum(w[tiles])


def get_active_tiles(hash_table, numtilings, state, action, x_bound, v_bound):
    return tiles(hash_table, numtilings, normalize_state(state, numtilings, x_bound, v_bound), [action])


def normalize_state(state, numtilings, x_bound, v_bound):
    x = state[0] * (numtilings / (x_bound[1] - x_bound[0]))
    v = state[1] * (numtilings / (v_bound[1] - v_bound[0]))
    return [x, v]


def evaluate(env, w, hashtable, numtilings, actions, x_bound, v_bound, num_eval_episodes, debug=False):
    steps_list = []

    for episode in range(num_eval_episodes):
        state = env.reset()[0]
        steps = 0

        while True:
            action = greedy_choice(w, state, numtilings, actions, hashtable, x_bound, v_bound)

            new_state, reward, terminated, truncated, info = env.step(action)

            # If done, finish the episode
            if terminated:
                steps_list.append(steps)
                if debug and (episode + 1) % 10 == 0:
                    print(f"Evaluation episode: {episode} - steps: {steps}")
                break

            steps += 1
            state = new_state

    return steps_list


def train(env, w, epsilon, numtilings, actions, hashtable, num_training_episodes, gamma, lmbda, x_bound, v_bound,
          learning_rate, debug=False):

    steps_list = []
    lr = learning_rate / numtilings

    for episode in range(num_training_episodes):
        state = env.reset()[0]
        z = np.zeros(w.shape)
        # epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        steps = 0

        while True:
            action = epsilon_greedy(env, epsilon, w, numtilings, actions, state, hashtable, x_bound, v_bound)

            new_state, reward, terminated, truncated, info = env.step(action)

            # Calculate current learning rate - lr decay
            # lr = max(min_lr, learning_rate * np.exp(-k * episode))

            active_tiles = get_active_tiles(hashtable, numtilings, state, action, x_bound, v_bound)

            # Update eligibility trace
            z = gamma * lmbda * z
            z[active_tiles] += 1

            delta = reward + gamma * (value(new_state, w, numtilings, action, hashtable, x_bound, v_bound)
                                      - value(state, w, numtilings, action, hashtable, x_bound, v_bound))

            w = w + lr * delta * z

            # If done, finish the episode
            if terminated:
                steps_list.append(steps)
                if debug and (episode+1) % 10 == 0:
                    print(f"Training episode: {episode}, lr: {lr}, lambda: {lmbda} - steps: {steps}")
                break

            steps += 1
            state = new_state

    return steps_list
