import numpy as np
import gymnasium as gym
# from tiling import get_active_tiles
from TileCoding import *
import matplotlib.pyplot as plt
from helper_TD_L import evaluate, train


env = gym.make('MountainCar-v0')

# debug
debug = True

maxsize = 2048
# w = np.zeros(20 * 20, 3)

numtilings = 8

x_bound = [-1.2, 0.6]
v_bound = [-0.07, 0.07]
actions = [0, 1, 2]

# Hyperparameters
lmbdas = [0.2, 0.4, 0.5, 0.7, 0.9]  # original: 0.5
gamma = 0.95

learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5]  # original: 0.1
# k = 0.005
# min_lr = 0.005

num_training_episodes = 400


# Evaluation Parameters
eval_runs = 100
num_eval_episodes = 100

# num_actions = env.action_space.n
# num_obs = env.observation_space.shape[0]

# Exploration parameters
# max_epsilon = 0.25   # Exploration probability at start                     # old: 1
# min_epsilon = 0.05  # Minimum exploration probability
# decay_rate = 0.005  # Exponential decay rate for exploration prob           # old: 0.002
epsilon = 0.05

# Visualization parameters
fig_size = 50
font_x_y = 36
font_title = 50
# Create 5 x 5 plot for total steps
num_subplots = 5

fig, axs = plt.subplots(nrows=num_subplots, ncols=num_subplots, figsize=(fig_size, fig_size))


for i, learning_rate in enumerate(learning_rates):
    for j, lmbda in enumerate(lmbdas):
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
                print(f'Average number of steps at {np.average(steps_list_eval)}')

        # print(f'Average number of steps: {total_steps / (eval_runs * num_eval_episodes)}')

        # Plot
        axs[i, j].plot(np.arange(len(steps_list)) + 1, steps_list)
        axs[i, j].title(f'Average evaluation steps: {np.average(steps_list_eval)}')
        axs[i, j].get_xaxis().set_visible(False)
        axs[i, j].get_yaxis().set_visible(False)


# fig.supxlabel('Learning rates\n0.001, 0.005, 0.01, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 2', fontsize=font_x_y)
# fig.supylabel('Gamma\n0, 0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.99, 1', fontsize=font_x_y)
fig.suptitle('Total steps needed per Episode based on different values for lambda and lr', fontsize=font_title)
plt.savefig('plots/total_steps_5x5.png')
plt.close()

env.close()
