import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from TileCoding import IHT
from helper_TD_L import evaluate, train


env = gym.make('MountainCar-v0')

# debug
debug = True

maxsize = 2048

numtilings = 8

x_bound = [-1.2, 0.6]
v_bound = [-0.07, 0.07]
actions = [0, 1, 2]

# Hyperparameters
lmbda = 0.5
gamma = 0.95

learning_rate = 0.1

num_training_episodes = 400
eval_runs = 100
num_eval_episodes = 100

epsilon = 0.05

w = np.zeros(maxsize)
hashtable = IHT(maxsize)

# Train
steps_list = train(env, w, epsilon, numtilings, actions, hashtable, num_training_episodes, gamma, lmbda,
                           x_bound, v_bound, learning_rate, True)

# Evaluate
total_steps = 0

for k in range(eval_runs):
    steps_list_eval = evaluate(env, w, hashtable, numtilings, actions, x_bound, v_bound, num_eval_episodes, True)
    total_steps += np.sum(steps_list_eval)
    if (k+1) % 10 == 0:
        print(f'Eval run: {k} - Average number of steps: {np.average(steps_list_eval)}')

print(f'Average number of steps: {total_steps / (eval_runs * num_eval_episodes)}')

# Plot Steps
plt.plot(np.arange(len(steps_list)) + 1, steps_list)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title(f'Steps per Episode')
plt.savefig(f'plots/steps_TD_L.png')
plt.close()
