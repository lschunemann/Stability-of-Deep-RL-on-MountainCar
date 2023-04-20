from helper import evaluate, initialize_grids, initialize_state_dict
import gymnasium as gym
import numpy as np


env = gym.make("MountainCar-v0")

grid_x, grid_v = initialize_grids()

state_to_qtable = initialize_state_dict()

# Load Q table to be evaluated
q_main = np.loadtxt('data/q_car.txt')
q_rand = np.loadtxt('data/q_random.txt')
q_decay = np.loadtxt('data/q_rand_decay.txt')
qs = [(q_main, 'main'), (q_rand, 'rand'), (q_decay, 'decay')]

# Environment parameters
max_steps = int(10000)  # Max steps per episode
eval_seed = []  # The evaluation seed of the environment

n_eval_episodes = 100  # Total number of test episodes
trials = 1000

total_main = []
total_rand = []
total_decay = []


for i in range(trials):
    for q, name in qs:
        mean_reward, std_reward = evaluate(env, max_steps, n_eval_episodes, q, grid_x, grid_v, state_to_qtable, eval_seed)
        if name == 'main':
            total_main.append(mean_reward)
        elif name == 'rand':
            total_rand.append(mean_reward)
        else:
            total_decay.append(mean_reward)
        # Print current mean rewards every 10 trials
        if (i+1) % 10 == 0:
            print(f"Trial: {i} - Model: {name}\t Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

# Print average rewards over all trials
print(f'Average rewards for main: {np.average(total_main)}')
print(f'Average rewards for rand: {np.average(total_rand)}')
print(f'Average rewards for decay: {np.average(total_decay)}')
