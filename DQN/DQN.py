import torch
import numpy as np
from TrainMountainCar import TrainMountainCar
import matplotlib.pyplot as plt
from helper_DQN import running_mean

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device("cuda")

# Hyperparameters
learning_rate = 0.00025

# Exploration parameters
epsilon_max = 1
epsilon_min = 0.1
eval_epsilon = 0.05


car = TrainMountainCar(learning_rate=learning_rate,
                       epsilon_max=epsilon_max, epsilon_min=epsilon_min, debug=True,
                       eval_epsilon=eval_epsilon)

total_rewards, total_steps_list, q_measures, best_policy, evaluations, td_error, final_policy = car.train()

# save best policy as well as steps and q measures
torch.save(best_policy, 'data/DQN_best.pth')
torch.save(final_policy, 'data/DQN_final.pth')
np.savetxt('data/td_error_DQN.txt', td_error)
np.savetxt(f'data/steps_DQN.txt', total_steps_list)
np.savetxt(f'data/q_values_DQN.txt', q_measures)
np.savetxt(f'data/eval_DQN.txt', evaluations)


# Plot steps per episode
plt.plot(np.arange(len(total_steps_list)) + 1, total_steps_list, zorder=0, label='training')
x = np.arange(50, 1001, 50)
plt.scatter(x, [-e*4 for e in evaluations], color='r', marker='x', zorder=1, label='evaluations')
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
