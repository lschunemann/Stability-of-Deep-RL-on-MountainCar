import gymnasium as gym
from helper import initialize_grids, initialize_state_dict, greedy_policy
import numpy as np

env = gym.make("MountainCar-v0", render_mode="human")

state = env.reset()
max_steps = 10000

grid_x, grid_v = initialize_grids()
state_to_qtable = initialize_state_dict()

q = np.loadtxt('data/q_rand_decay.txt')

for step in range(max_steps):
    # Take the action (index) that have the maximum expected future reward given that state
    action = greedy_policy(q, state, grid_x, grid_v, state_to_qtable)
    new_state, reward, terminated, truncated, info = env.step(action)

    if terminated:  # or truncated:
        break
    state = new_state
