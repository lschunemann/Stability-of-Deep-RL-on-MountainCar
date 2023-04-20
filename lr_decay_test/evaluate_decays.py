from standard.helper import evaluate, initialize_grids, initialize_state_dict
import gymnasium as gym
import numpy as np


env = gym.make("MountainCar-v0")

grid_x, grid_v = initialize_grids()

state_to_qtable = initialize_state_dict()

# Load Q table to be evaluated
q_1 = np.loadtxt('data/q_rand_decay_1_0.01.txt')
q_2 = np.loadtxt('data/q_rand_decay_0.5_0.01.txt')
q_3 = np.loadtxt('data/q_rand_decay_0.5_0.005.txt')
q_4 = np.loadtxt('data/q_rand_decay_0.25_0.001.txt')
q_5 = np.loadtxt('data/q_rand_decay_step_100eps.txt')
q_6 = np.loadtxt('data/q_rand_decay_steps_100_500eps.txt')
q_7 = np.loadtxt('data/q_rand_decay_steps_100_500_750eps.txt')
q_8 = np.loadtxt('data/q_rand_decay_fixed.txt')
qs = [(q_1, '1_0.01'), (q_2, '0.5_0.01'), (q_3, '0.5_0.005'), (q_4, '0.25_0.001'),
      (q_6, 'steps_100_500eps'), (q_7, 'steps_100_500_750eps'), (q_8, 'fixed')]

# Environment parameters
max_steps = int(10000)  # Max steps per episode
eval_seed = []  # The evaluation seed of the environment

n_eval_episodes = 100  # Total number of test episodes
trials = 100

total_1 = []
total_2 = []
total_3 = []
total_4 = []
# total_step_100 = []
total_steps_100_500 = []
total_steps_100_500_750 = []
total_fixed = []

file = open('evaluations_decay.txt', 'w')

for i in range(trials):
    for q, name in qs:
        mean_reward, std_reward = evaluate(env, max_steps, n_eval_episodes, q, grid_x, grid_v, state_to_qtable, eval_seed)
        if name == '1_0.01':
            total_1.append(mean_reward)
        elif name == '0.5_0.01':
            total_2.append(mean_reward)
        elif name == '0.5_0.005':
            total_3.append(mean_reward)
        elif name == '0.25_0.001':
            total_4.append(mean_reward)
        # elif name == 'step_100':
        #     total_step_100.append(mean_reward)
        elif name == 'steps_100_500':
            total_steps_100_500.append(mean_reward)
        elif name == 'steps_100_500_750':
            total_steps_100_500_750.append(mean_reward)
        elif name == 'fixed':
            total_fixed.append(mean_reward)
        # Print current mean rewards every 10 trials
        print(f"Trial: {i} - Model: {name}\t Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
        if (i+1) % 10 == 0:
            file.write(f"Trial: {i} - Model: {name}\t Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}\n")

# Print average rewards over all trials
file.write(f'Average rewards for 1_0.01: {np.average(total_1)}\n')
file.write(f'Average rewards for 0.5_0.01: {np.average(total_2)}\n')
file.write(f'Average rewards for 0.5_0.005: {np.average(total_3)}\n')
file.write(f'Average rewards for 0.25_0.001: {np.average(total_4)}\n')
# file.write(f'Average rewards for step_100: {np.average(total_step_100)}')
file.write(f'Average rewards for steps_100_500: {np.average(total_steps_100_500)}\n')
file.write(f'Average rewards for steps_100_500_750: {np.average(total_steps_100_500_750)}\n')
file.write(f'Average rewards for fixed: {np.average(total_fixed)}\n')
file.close()

env.close()
