import torch
from TrainMountainCar import TrainMountainCar
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda")

# Hyperparameters
n_training_episodes = 500
gamma = 0.99
# learning_rate = 0.00025  # 0.1
learning_rate = 0.0000625
max_training_steps = 10000

# Exploration parameters
epsilon_max = 1
epsilon_min = 0.01
decay_rate = 0.00001

# replay memory parameters
replay_size = 100000
batch_size = 32


# fixed target network
fixed_target = True
copy_target = 30000


debug = True
double = True       # DDQN
dueling = True      # Dueling Network
prioritized = True  # Prioritized Experience Replay

car = TrainMountainCar(n_training_episodes=n_training_episodes, gamma=gamma, learning_rate=learning_rate,
                       epsilon_max=epsilon_max, epsilon_min=epsilon_min, decay_rate=decay_rate,
                       max_steps=max_training_steps, batch_size=batch_size, fixed_target=fixed_target,
                       copy_target=copy_target, replay_size=replay_size, double=double, dueling=dueling, debug=debug,
                       prioritized=prioritized)

total_rewards, total_steps_list, q_measures, best_policy = car.train()

torch.save(best_policy, 'data/Prioritized_Dueling_DDQN.pth')
np.savetxt(f'data/steps_Prioritized_Dueling_DDQN.txt', total_steps_list)
np.savetxt(f'data/q_values_Prioritized_Dueling_DDQN.txt', q_measures)

plt.plot(np.arange(len(total_steps_list)) + 1, total_steps_list)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode - Prioritized_Dueling_DDQN')
plt.savefig('plots/steps_Prioritized_Dueling_DDQN.png')
plt.close()

plt.plot(np.arange(len(q_measures)) + 1, q_measures)
plt.xlabel('Episode')
plt.ylabel('Average Q')
plt.title('Average Q measure over sampled states')
plt.savefig('plots/q_measures_Prioritized_Dueling_DDQN.png')
plt.close()

