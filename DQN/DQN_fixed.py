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
double = False

car = TrainMountainCar(learning_rate=learning_rate,
                       epsilon_max=epsilon_max, epsilon_min=epsilon_min,
                       fixed_target=fixed_target,
                       copy_target=copy_target, double=double, debug=debug, eval_epsilon=eval_epsilon)

total_rewards, total_steps_list, q_measures, best_policy, evaluations, td_error, final_policy = car.train()

# save best policy as well as steps and q measures
torch.save(best_policy, 'data/DQN_paper_fixed_best.pth')
torch.save(final_policy, 'data/DQN_paper_fixed_final.pth')
np.savetxt(f'data/steps_fixed.txt', total_steps_list)
np.savetxt(f'data/q_values_fixed.txt', q_measures)
np.savetxt(f'data/eval_fixed.txt', evaluations)
np.savetxt(f'data/td_error_Distributional_DDQN.txt', td_error)

# Plot steps per episode
plt.plot(np.arange(len(total_steps_list)) + 1, total_steps_list, zorder=0, label='training')
x = np.arange(50, 1000+1, 50)
plt.scatter(x, [-e*4 for e in evaluations], color='r', marker='x', zorder=1, label='evaluations')
N = 10
steps_mean = running_mean(total_steps_list, N)
plt.plot(np.arange(len(steps_mean)) + 1, steps_mean, zorder=0, label='running average')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode - DQN_fixed')
plt.savefig('plots/steps_DQN_fixed.png')
plt.close()

# Plot q measures per episode
plt.plot(np.arange(len(q_measures)) + 1, q_measures)
plt.xlabel('Episode')
plt.ylabel('Average Q')
plt.title('Average Q measure over sampled states')
plt.savefig('plots/q_measures_DQN_fixed.png')
plt.close()
