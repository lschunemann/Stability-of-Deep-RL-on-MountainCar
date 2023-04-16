import numpy as np
import gymnasium as gym
# from train import getActiveTiles, epsilon_greedy
from TileCoding import *

env = gym.make('MountainCar-v0')

# for i in range(3):
#     w = (i+1) * ((1 / 20) / 3) * np.asarray([1, 3])
#     print(w)
# maxsize = 2048
# numtilings = 8
# epsilon = 0.05
# w = np.zeros(maxsize)
# state = env.reset()
# action = epsilon_greedy(env, w, epsilon)
# x_bound = [-1.2, 0.6]
# v_bound = [-0.07, 0.07]
# print(state)
# print(state[0] * (numtilings / x_bound[1] - x_bound[0]))


def normalize_state(state, numtilings):
    x = state[0] * (numtilings / (x_bound[1] - x_bound[0]))
    v = state[1] * (numtilings / (v_bound[1] - v_bound[0]))
    return [x, v]


x_bound = [-1.2, 0.6]
v_bound = [-0.07, 0.07]

maxsize = 2048
w = np.zeros(maxsize)
actions = [-1, 0, 1]
hash_table = IHT(maxsize)
numtilings = 8
z = np.zeros(w.shape)

state = env.reset()[0]

for action in actions:
    print(tiles(hash_table, numtilings, normalize_state(state, numtilings), [action]))
    # new_state, reward, terminated, truncated, info = env.step(action)
    # print(tiles(hash_table, numtilings, normalize_state(new_state, numtilings), [action]))

for i in range(100):
    action = env.action_space.sample()
    print(tiles(hash_table, numtilings, normalize_state(state, numtilings), [action]))
