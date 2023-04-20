from standard.helper import evaluate, initialize_grids, initialize_state_dict
import gymnasium as gym
import numpy as np


env = gym.make("MountainCar-v0")

grid_x, grid_v = initialize_grids()

state_to_qtable = initialize_state_dict()

lrs = [(1, 0.01), (0.5, 0.005), (0.5, 0.01), (0.25, 0.005), (0.25, 0.001)]
decays = [(1, 0.005), (0.5, 0.005), (0.5, 0.01), (0.25, 0.005), (0.25, 0.01)]

# Load Q table to be evaluated
q1 = np.loadtxt("data/q_lr_1,0.01_eps_1,0.005.txt")
q2 = np.loadtxt("data/q_lr_1,0.01_eps_0.5,0.005.txt")
q3 = np.loadtxt("data/q_lr_1,0.01_eps_0.5,0.01.txt")
q4 = np.loadtxt("data/q_lr_1,0.01_eps_0.25,0.005.txt")
q5 = np.loadtxt("data/q_lr_1,0.01_eps_0.25,0.01.txt")
q6 = np.loadtxt("data/q_lr_0.5,0.005_eps_1,0.005.txt")
q7 = np.loadtxt("data/q_lr_0.5,0.005_eps_0.5,0.005.txt")
q8 = np.loadtxt("data/q_lr_0.5,0.005_eps_0.5,0.01.txt")
q9 = np.loadtxt("data/q_lr_0.5,0.005_eps_0.25,0.005.txt")
q10 = np.loadtxt("data/q_lr_0.5,0.005_eps_0.25,0.01.txt")
q11 = np.loadtxt("data/q_lr_0.5,0.01_eps_1,0.005.txt")
q12 = np.loadtxt("data/q_lr_0.5,0.01_eps_0.5,0.005.txt")
q13 = np.loadtxt("data/q_lr_0.5,0.01_eps_0.5,0.01.txt")
q14 = np.loadtxt("data/q_lr_0.5,0.01_eps_0.25,0.005.txt")
q15 = np.loadtxt("data/q_lr_0.5,0.01_eps_0.25,0.01.txt")
q16 = np.loadtxt("data/q_lr_0.25,0.005_eps_1,0.005.txt")
q17 = np.loadtxt("data/q_lr_0.25,0.005_eps_0.5,0.005.txt")
q18 = np.loadtxt("data/q_lr_0.25,0.005_eps_0.5,0.01.txt")
q19 = np.loadtxt("data/q_lr_0.25,0.005_eps_0.25,0.005.txt")
q20 = np.loadtxt("data/q_lr_0.25,0.005_eps_0.25,0.01.txt")
q21 = np.loadtxt("data/q_lr_0.25,0.001_eps_1,0.005.txt")
q22 = np.loadtxt("data/q_lr_0.25,0.001_eps_0.5,0.005.txt")
# q23 = np.loadtxt("data/q_lr_0.25,0.001_eps_0.5,0.01.txt")
q24 = np.loadtxt("data/q_lr_0.25,0.001_eps_0.25,0.005.txt")
q25 = np.loadtxt("data/q_lr_0.25,0.001_eps_0.25,0.01.txt")

qs = [(q1, 'q_lr_1,0.01_eps_1,0.005'), (q2, 'q_lr_1,0.01_eps_0.5,0.005'), (q3, 'q_lr_1,0.01_eps_0.5,0.01'),
 (q4, 'q_lr_1,0.01_eps_0.25,0.005'), (q5, 'q_lr_1,0.01_eps_0.25,0.01'), (q6, 'q_lr_0.5,0.005_eps_1,0.005'),
 (q7, 'q_lr_0.5,0.005_eps_0.5,0.005'), (q8, 'q_lr_0.5,0.005_eps_0.5,0.01'), (q9, 'q_lr_0.5,0.005_eps_0.25,0.005'),
 (q10, 'q_lr_0.5,0.005_eps_0.25,0.01'), (q11, 'q_lr_0.5,0.01_eps_1,0.005'), (q12, 'q_lr_0.5,0.01_eps_0.5,0.005'),
 (q13, 'q_lr_0.5,0.01_eps_0.5,0.01'), (q14, 'q_lr_0.5,0.01_eps_0.25,0.005'), (q15, 'q_lr_0.5,0.01_eps_0.25,0.01'),
 (q16, 'q_lr_0.25,0.005_eps_1,0.005'), (q17, 'q_lr_0.25,0.005_eps_0.5,0.005'),
 (q18, 'q_lr_0.25,0.005_eps_0.5,0.01'), (q19, 'q_lr_0.25,0.005_eps_0.25,0.005'),
 (q20, 'q_lr_0.25,0.005_eps_0.25,0.01'), (q21, 'q_lr_0.25,0.001_eps_1,0.005'),
 (q22, 'q_lr_0.25,0.001_eps_0.5,0.005'),  # (q23, 'q_lr_0.25,0.001_eps_0.5,0.01'),
 (q24, 'q_lr_0.25,0.001_eps_0.25,0.005'), (q25, 'q_lr_0.25,0.001_eps_0.25,0.01')]

# Environment parameters
max_steps = 10000  # Max steps per episode
eval_seed = []  # The evaluation seed of the environment

n_eval_episodes = 100  # Total number of test episodes
trials = 100

total_1 = []
total_2 = []
total_3 = []
total_4 = []
total_5 = []
total_6 = []
total_7 = []
total_8 = []
total_9 = []
total_10 = []
total_11 = []
total_12 = []
total_13 = []
total_14 = []
total_15 = []
total_16 = []
total_17 = []
total_18 = []
total_19 = []
total_20 = []
total_21 = []
total_22 = []
# total_23 = []
total_24 = []
total_25 = []


file = open('evaluations_lr_eps.txt', 'w')

for i in range(trials):
    for q, name in qs:
        mean_reward, std_reward = evaluate(env, max_steps, n_eval_episodes, q, grid_x, grid_v,
                                                      state_to_qtable, eval_seed)
        if name == "q_lr_1,0.01_eps_1,0.005":
            total_1.append(mean_reward)
        elif name == "q_lr_1,0.01_eps_0.5,0.005":
            total_2.append(mean_reward)
        elif name == "q_lr_1,0.01_eps_0.5,0.01":
            total_3.append(mean_reward)
        elif name == "q_lr_1,0.01_eps_0.25,0.005":
            total_4.append(mean_reward)
        elif name == "q_lr_1,0.01_eps_0.25,0.01":
            total_5.append(mean_reward)
        elif name == "q_lr_0.5,0.005_eps_1,0.005":
            total_6.append(mean_reward)
        elif name == "q_lr_0.5,0.005_eps_0.5,0.005":
            total_7.append(mean_reward)
        elif name == "q_lr_0.5,0.005_eps_0.5,0.01":
            total_8.append(mean_reward)
        elif name == "q_lr_0.5,0.005_eps_0.25,0.005":
            total_9.append(mean_reward)
        elif name == "q_lr_0.5,0.005_eps_0.25,0.01":
            total_10.append(mean_reward)
        elif name == "q_lr_0.5,0.01_eps_1,0.005":
            total_11.append(mean_reward)
        elif name == "q_lr_0.5,0.01_eps_0.5,0.005":
            total_12.append(mean_reward)
        elif name == "q_lr_0.5,0.01_eps_0.5,0.01":
            total_13.append(mean_reward)
        elif name == "q_lr_0.5,0.01_eps_0.25,0.005":
            total_14.append(mean_reward)
        elif name == "q_lr_0.5,0.01_eps_0.25,0.01":
            total_15.append(mean_reward)
        elif name == "q_lr_0.25,0.005_eps_1,0.005":
            total_16.append(mean_reward)
        elif name == "q_lr_0.25,0.005_eps_0.5,0.005":
            total_17.append(mean_reward)
        elif name == "q_lr_0.25,0.005_eps_0.5,0.01":
            total_18.append(mean_reward)
        elif name == "q_lr_0.25,0.005_eps_0.25,0.005":
            total_19.append(mean_reward)
        elif name == "q_lr_0.25,0.005_eps_0.25,0.01":
            total_20.append(mean_reward)
        elif name == "q_lr_0.25,0.001_eps_1,0.005":
            total_21.append(mean_reward)
        elif name == "q_lr_0.25,0.001_eps_0.5,0.005":
             total_22.append(mean_reward)
        # elif name == "q_lr_0.25,0.001_eps_0.5,0.01":
        #     total_23.append(mean_reward)
        elif name == "q_lr_0.25,0.001_eps_0.25,0.005":
            total_24.append(mean_reward)
        elif name == "q_lr_0.25,0.001_eps_0.25,0.01":
            total_25.append(mean_reward)
        # Print current mean rewards every 10 trials
        print(f"Trial: {i} - Model: {name}\t Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
        if (i+1) % 10 == 0:
            file.write(f"Trial: {i} - Model: {name}\t Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}\n")

env.close()

# Print average rewards over all trials
file.write('\n')
file.write(f'Average rewards for lr:1,0.01_eps:1,0.005: {np.average(total_1)}')
file.write(f'Average rewards for lr:1,0.01_eps:0.5,0.005: {np.average(total_2)}')
file.write(f'Average rewards for lr:1,0.01_eps:0.5,0.01: {np.average(total_3)}')
file.write(f'Average rewards for lr:1,0.01_eps:0.25,0.005: {np.average(total_4)}')
file.write(f'Average rewards for lr:1,0.01_eps:0.25,0.01: {np.average(total_5)}')
file.write(f'Average rewards for lr:0.5,0.005_eps:1,0.005: {np.average(total_6)}')
file.write(f'Average rewards for lr:0.5,0.005_eps:0.5,0.005: {np.average(total_7)}')
file.write(f'Average rewards for lr:0.5,0.005_eps:0.5,0.01: {np.average(total_8)}')
file.write(f'Average rewards for lr:0.5,0.005_eps:0.25,0.005: {np.average(total_9)}')
file.write(f'Average rewards for lr:0.5,0.005_eps:0.25,0.01: {np.average(total_10)}')
file.write(f'Average rewards for lr:0.5,0.01_eps:1,0.005: {np.average(total_11)}')
file.write(f'Average rewards for lr:0.5,0.01_eps:0.5,0.005: {np.average(total_12)}')
file.write(f'Average rewards for lr:0.5,0.01_eps:0.5,0.01: {np.average(total_13)}')
file.write(f'Average rewards for lr:0.5,0.01_eps:0.25,0.005: {np.average(total_14)}')
file.write(f'Average rewards for lr:0.5,0.01_eps:0.25,0.01: {np.average(total_15)}')
file.write(f'Average rewards for lr:0.25,0.005_eps:1,0.005: {np.average(total_16)}')
file.write(f'Average rewards for lr:0.25,0.005_eps:0.5,0.005: {np.average(total_17)}')
file.write(f'Average rewards for lr:0.25,0.005_eps:0.5,0.01: {np.average(total_18)}')
file.write(f'Average rewards for lr:0.25,0.005_eps:0.25,0.005: {np.average(total_19)}')
file.write(f'Average rewards for lr:0.25,0.005_eps:0.25,0.01: {np.average(total_20)}')
file.write(f'Average rewards for lr:0.25,0.001_eps:1,0.005: {np.average(total_21)}')
file.write(f'Average rewards for lr:0.25,0.001_eps:0.5,0.005: {np.average(total_22)}')
# file.write(f'Average rewards for lr:0.25,0.001_eps:0.5,0.01: {np.average(total_23)}')
file.write(f'Average rewards for lr:0.25,0.001_eps:0.25,0.005: {np.average(total_24)}')
file.write(f'Average rewards for lr:0.25,0.001_eps:0.25,0.01: {np.average(total_25)}')
file.close()
