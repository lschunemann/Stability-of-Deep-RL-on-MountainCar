import torch
from TrainMountainCar import TrainMountainCar
import numpy as np
import matplotlib.pyplot as plt
from helper_DQN import running_mean

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# Hyperparameters
n_training_episodes = 1000
gamma = 0.99
learning_rate = 0.00005  # 0.1
max_training_steps = 10000

# Exploration parameters
epsilon_max = 1
epsilon_min = 0.01
eval_epsilon = 0.001

# replay memory parameters
replay_size = 100000
batch_size = 32


# fixed target network
fixed_target = True
copy_target = 10000


debug = True
quantile = True
quantiles = 51

car = TrainMountainCar(n_training_episodes=n_training_episodes, gamma=gamma, learning_rate=learning_rate,
                       epsilon_max=epsilon_max, epsilon_min=epsilon_min,
                       max_steps=max_training_steps, batch_size=batch_size, fixed_target=fixed_target,
                       quantiles=quantiles, quantile=True, copy_target=copy_target, replay_size=replay_size,
                       debug=debug, eval_epsilon=eval_epsilon)

total_rewards, total_steps_list, q_measures, best_policy, evaluations, td_error, final_policy = car.train()

# Save best policy, as well as steps and q measures
torch.save(best_policy, 'data/Quantile_DDQN_best.pth')
torch.save(final_policy, 'data/Quantile_DDQN_final.pth')
np.savetxt(f'data/steps_Quantile_DDQN.txt', total_steps_list)
np.savetxt(f'data/q_values_Quantile_DDQN.txt', q_measures)
np.savetxt(f'data/eval_Quantile_DDQN.txt', evaluations)
np.savetxt(f'data/td_error_Quantile_DDQN.txt', td_error)

# Plot steps over episodes
plt.plot(np.arange(len(total_steps_list)) + 1, total_steps_list, zorder=0, label='training')
x = np.arange(50, n_training_episodes+1, 50)
plt.scatter(x, [-e*4 for e in evaluations], color='r', marker='x', zorder=1, label='evaluations')
N = 10
steps_mean = running_mean(total_steps_list, N)
plt.plot(np.arange(len(steps_mean)) + 1, steps_mean, zorder=0, label='running average')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode - Quantile_DDQN')
plt.savefig('plots/steps_Quantile_DDQN.png')
plt.close()

# Plot Q measures over episodes
plt.plot(np.arange(len(q_measures)) + 1, q_measures)
plt.xlabel('Episode')
plt.ylabel('Average Q')
plt.title('Average Q measure over sampled states')
plt.savefig('plots/q_measures_Quantile_DDQN.png')
plt.close()
