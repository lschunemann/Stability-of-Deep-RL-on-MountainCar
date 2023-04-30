import torch
from TrainMountainCar import TrainMountainCar
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda")

# Hyperparameters
n_training_episodes = 500
gamma = 0.99
learning_rate = 0.00025  # 0.1
max_training_steps = 10000

# Exploration parameters
epsilon_max = 1
epsilon_min = 0.01

# replay memory parameters
replay_size = 100000
batch_size = 32


# fixed target network
fixed_target = True
copy_target = 30000


debug = True
double = True
noisy = True

car = TrainMountainCar(n_training_episodes=n_training_episodes, gamma=gamma, learning_rate=learning_rate,
                       epsilon_max=epsilon_max, epsilon_min=epsilon_min,
                       max_steps=max_training_steps, batch_size=batch_size, fixed_target=fixed_target,
                       copy_target=copy_target, replay_size=replay_size, double=double, debug=debug, noisy=noisy)

total_rewards, total_steps_list, q_measures, best_policy, evaluations = car.train()

# Save best policy, as well as steps and q measures
torch.save(best_policy, 'data/Noisy_DDQN.pth')
np.savetxt(f'data/steps_Noisy_DDQN.txt', total_steps_list)
np.savetxt(f'data/q_values_Noisy_DDQN.txt', q_measures)

# Plot steps over episodes
plt.plot(np.arange(len(total_steps_list)) + 1, total_steps_list)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode - Noisy_DDQN')
plt.savefig('plots/steps_Noisy_DDQN.png')
plt.close()

# Plot Q measures over episodes
plt.plot(np.arange(len(q_measures)) + 1, q_measures)
plt.xlabel('Episode')
plt.ylabel('Average Q')
plt.title('Average Q measure over sampled states')
plt.savefig('plots/q_measures_Noisy_DDQN.png')
plt.close()
