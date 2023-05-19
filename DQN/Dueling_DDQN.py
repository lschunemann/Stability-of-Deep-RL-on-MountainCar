import torch
from TrainMountainCar import TrainMountainCar
import numpy as np
import matplotlib.pyplot as plt
from helper_DQN import running_mean

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device("cuda")

# Hyperparameters
n_training_episodes = 1000
gamma = 0.99
learning_rate = 0.00025  # 0.1
max_training_steps = 10000

# Exploration parameters
epsilon_max = 1
epsilon_min = 0.1
eval_epsilon = 0.05

# replay memory parameters
replay_size = 200000
batch_size = 32
min_memory = 80000


# fixed target network
fixed_target = True
copy_target = 30000


debug = True
double = True       # DDQN
dueling = True      # Dueling Network

car = TrainMountainCar(n_training_episodes=n_training_episodes, gamma=gamma, learning_rate=learning_rate,
                       epsilon_max=epsilon_max, epsilon_min=epsilon_min, min_memory=min_memory, eval_epsilon=eval_epsilon,
                       max_steps=max_training_steps, batch_size=batch_size, fixed_target=fixed_target,
                       copy_target=copy_target, replay_size=replay_size, double=double, dueling=dueling, debug=debug)

total_rewards, total_steps_list, q_measures, best_policy, evaluations, td_error, final_policy = car.train()

torch.save(best_policy, 'data/Dueling_DDQN_best.pth')
torch.save(final_policy, 'data/Dueling_DDQN_final.pth')
np.savetxt(f'data/steps_Dueling_DDQN.txt', total_steps_list)
np.savetxt(f'data/q_values_Dueling_DDQN.txt', q_measures)
np.savetxt(f'data/eval_Dueling_DDQN.txt', evaluations)
np.savetxt(f'data/td_error_Dueling_DDQN.txt', td_error)

plt.plot(np.arange(len(total_steps_list)) + 1, total_steps_list, zorder=0, label='training')
x = np.arange(50, n_training_episodes, 50)
plt.scatter(x, [-e*4 for e in evaluations], color='r', marker='x', zorder=1, label='evaluations')
N = 10
steps_mean = running_mean(total_steps_list, N)
plt.plot(np.arange(len(steps_mean)) + 1, steps_mean, zorder=0, label='running average')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode - Dueling_DDQN')
plt.savefig('plots/steps_Dueling_DDQN.png')
plt.close()

plt.plot(np.arange(len(q_measures)) + 1, q_measures)
plt.xlabel('Episode')
plt.ylabel('Average Q')
plt.title('Average Q measure over sampled states')
plt.savefig('plots/q_measures_Dueling_DDQN.png')
plt.close()
