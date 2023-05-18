import torch
import numpy as np
from TrainMountainCar import TrainMountainCar
import matplotlib.pyplot as plt
from helper_DQN import running_mean

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device("cuda")

# Hyperparameters
n_training_episodes = 1000
gamma = 0.99
learning_rate = 0.00025
max_training_steps = 10000

# Exploration parameters
epsilon_max = 1
epsilon_min = 0.1
eval_epsilon = 0.05

# replay memory parameters
replay_size = 200000
batch_size = 32
min_memory = 80000

car = TrainMountainCar(n_training_episodes=n_training_episodes, gamma=gamma, learning_rate=learning_rate,
                       epsilon_max=epsilon_max, epsilon_min=epsilon_min, replay_size=replay_size,
                       max_steps=max_training_steps, batch_size=batch_size, debug=True, min_memory=min_memory,
                       eval_epsilon=eval_epsilon)

total_rewards, total_steps_list, q_measures, best_policy, evaluations = car.train()

# save best policy as well as steps and q measures
torch.save(best_policy, 'data/DQN.pth')
np.savetxt(f'data/steps_DQN.txt', total_steps_list)
np.savetxt(f'data/q_values_DQN.txt', q_measures)
np.savetxt(f'data/eval_DQN.txt', evaluations)


# Plot steps per episode
plt.plot(np.arange(len(total_steps_list)) + 1, total_steps_list, zorder=0, label='training')
plt.scatter([50, 100, 150, 200, 250, 300, 350, 400, 450, 500], -evaluations, color='r', marker='x', zorder=1, label='evaluations')
N = 10
steps_mean = running_mean(total_steps_list, N)
plt.plot(np.arange(len(steps_mean)) + 1, steps_mean, zorder=0, label='running average')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode - DQN')
plt.savefig('plots/steps_DQN.png')
plt.close()

# Plot q measures per episode
plt.plot(np.arange(len(q_measures)) + 1, q_measures)
plt.xlabel('Episode')
plt.ylabel('Average Q')
plt.title('Average Q measure over sampled states')
plt.savefig('plots/q_measures_DQN.png')
plt.close()
