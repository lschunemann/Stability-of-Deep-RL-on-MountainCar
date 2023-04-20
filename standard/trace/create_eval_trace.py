import numpy as np
import gymnasium as gym
from standard.helper import greedy_policy, initialize_grids, initialize_state_dict

# Evaluation Parameters
seed = []
n_eval_episodes = 1
max_steps = 10000

# load q table
q = np.loadtxt('../data/q_rand_decay.txt')

# initialize discretization
grid_x, grid_v = initialize_grids()
state_to_qtable = initialize_state_dict()

# trace
file = open('trace_eval.txt', 'w')

env = gym.make('MountainCar-v0')

for episode in range(n_eval_episodes):
    if seed:
        state = env.reset(seed=seed[episode])
    else:
        state = env.reset()[0]
    step = 0
    terminated = False
    total_rewards_ep = 0

    for step in range(max_steps):
        action = greedy_policy(q, state, grid_x, grid_v, state_to_qtable)

        # write current state and action taken to trace
        file.write(f'{state[0]},{state[1]},{action}\n')

        new_state, reward, terminated, truncated, info = env.step(action)
        total_rewards_ep += reward

        if terminated:  # or truncated:
            break
        state = new_state


