import torch
from TrainMountainCar import TrainMountainCar
import numpy as np
import matplotlib.pyplot as plt
from helper_DQN import running_mean

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device("cuda")

# Hyperparameters
learning_rate = 0.00025  # 0.1

# Exploration parameters
epsilon_max = 1
epsilon_min = 0.1
eval_epsilon = 0.05

# fixed target network
fixed_target = True
copy_target = 10000


debug = True
double = True

car = TrainMountainCar(learning_rate=learning_rate,
                       epsilon_max=epsilon_max, epsilon_min=epsilon_min,
                       fixed_target=fixed_target,
                       copy_target=copy_target, double=double, debug=debug)


total_rewards, total_steps_list, q_measures, best_policy, evaluations, td_error, policy_final = car.train()

# Save best policy, as well as steps and q measures
torch.save(best_policy, 'data/DDQN_best.pth')
torch.save(policy_final, 'data/DDQN_final.pth')
np.savetxt(f'data/steps_DDQN.txt', total_steps_list)
np.savetxt(f'data/q_values_DDQN.txt', q_measures)
np.savetxt(f'data/eval_DDQN.txt', evaluations)
np.savetxt(f'data/td_error_DDQN.txt', td_error)

# Plot steps over episodes
plt.plot(np.arange(len(total_steps_list)) + 1, total_steps_list, label='training')
x = np.arange(50, 1001, 50)
plt.scatter(x, [-e*4 for e in evaluations], color='r', marker='x', zorder=1, label='evaluations')
N = 10
steps_mean = running_mean(total_steps_list, N)
plt.plot(np.arange(len(steps_mean)) + 1, steps_mean, zorder=0, label='running average')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode - DDQN')
plt.savefig('plots/steps_DDQN.png')
plt.close()

# Plot Q measures over episodes
plt.plot(np.arange(len(q_measures)) + 1, q_measures)
plt.xlabel('Episode')
plt.ylabel('Average Q')
plt.title('Average Q measure over sampled states')
plt.savefig('plots/q_measures_DDQN.png')
plt.close()
