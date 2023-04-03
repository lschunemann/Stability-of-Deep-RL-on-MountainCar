import random
import numpy as np


def initialize_grids():
    # initialize bins for x-axis and velocity
    grid_x = []
    grid_v = []
    for i in range(20):
        grid_x.append(round(-1.2 + i * 0.1, 2))
        grid_v.append(round(-0.07 + i * 0.008, 3))
    return grid_x, grid_v


def initialize_state_dict():
    state_to_qtable = {}
    x = -1.2
    v = -0.07
    i = 0
    while x <= 0.6:
        state_to_qtable[(x, v)] = i
        i += 1
        while v <= 0.07:
            state_to_qtable[(x, v)] = i
            v += 0.008
            v = round(v, 3)  # rounding to avoid floating point errors
            i += 1
        v = -0.07
        x += 0.1
        x = round(x, 2)  # rounding to avoid floating point errors
    return state_to_qtable


def get_closest_in_grid(n: np.ndarray, grid_x: list, grid_v: list) -> tuple:
    """round tuple n to the closest representation within the grid"""
    if type(n) != tuple:
        x = n[0]
        v = n[1]
    else:
        x = n[0][0]
        v = n[0][1]

    x = min(grid_x, key=lambda a: abs(a-x))
    v = min(grid_v, key=lambda a: abs(a-v))

    return x, v


def initialize_q_table(state_space, action_space):
    q = np.zeros((state_space, action_space))
    return q


def epsilon_greedy_policy(q, state, epsilon, grid_x, grid_v, state_to_qtable, env):
    random_num = random.uniform(0, 1)
    if random_num > epsilon:
        s = state_to_qtable[get_closest_in_grid(state, grid_x, grid_v)]
        choice = np.argmax(q[s])
    else:
        choice = env.action_space.sample()  # Take a random action
    return choice


def greedy_policy(q, state, grid_x, grid_v, state_to_qtable):
    s = state_to_qtable[get_closest_in_grid(state, grid_x, grid_v)]
    choice = np.argmax(q[s])
    return choice
