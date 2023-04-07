import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from helper import initialize_grids, initialize_q_table, initialize_state_dict, train, plot_rewards, plot_steps


env = gym.make("MountainCar-v0")

# initialize size of state and action space
state_space = 20 * 20
action_space = env.action_space.n

observation, info = env.reset(seed=42)

# Training parameters
n_training_episodes = 1000  # Total training episodes

# Evaluation parameters
# n_eval_episodes = 100  # Total number of test episodes

# Environment parameters
max_steps = 1000  # Max steps per episode
eval_seed = []  # The evaluation seed of the environment

# Exploration parameters
max_epsilon = 1.0   # Exploration probability at start
min_epsilon = 0.05  # Minimum exploration probability
decay_rate = 0.002  # Exponential decay rate for exploration prob

# Experiment setup
gamma = [0, 0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.99, 1]  # Discounting rate
learning_rate = [0.001, 0.005, 0.01, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 2]  # Learning rate

grid_x, grid_v = initialize_grids()
state_to_qtable = initialize_state_dict()

for g in gamma:
    for lr in learning_rate:
        q_car = initialize_q_table(state_space, action_space)
        q_car, avg_rewards, total_steps = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env,
                                                max_steps, q_car, grid_x, grid_v, lr, state_to_qtable, g, True)

        # Save data
        np.savetxt(f'data/q_rand_{g}_{lr}.txt', q_car)
        np.array(avg_rewards)
        np.savetxt(f'data/avg_rewards_rand_{g}_{lr}.txt', avg_rewards)
        np.array(total_steps)
        np.savetxt(f'data/total_steps_rand_{g}_{lr}.txt', total_steps)

        # plot data
        plot_rewards(avg_rewards, f'rand_{g}_{lr}')
        plot_steps(total_steps, f'rand_{g}_{lr}')

        # Plot V values
        actions = np.max(q_car, axis=1)
        actions = actions.reshape((20, 20))
        plt.figure(figsize=(16, 12))
        sns.heatmap(actions, annot=True)
        plt.ylim(0, 20)
        plt.xlabel("Position", fontsize=20)
        plt.ylabel("Velocity", fontsize=20)
        plt.title(f"V scores for optimal action - random init - gamma:{g} - lr:{lr}", fontdict={'fontsize': 25})
        plt.savefig(f'plots/v_values_random_{g}_{lr}.png')
        plt.close()

        # Plot Q table heatmap
        plt.figure(figsize=(12, 12))
        cmap = plt.cm.get_cmap('Greens', 3)
        plt.pcolor(range(0, 20), range(0, 20), actions, cmap=cmap)
        plt.xlabel("Position", fontsize=20)
        plt.ylabel("Velocity", fontsize=20)
        plt.title(f"Optimal action at state - random init - gamma:{g} - lr:{lr}", fontdict={'fontsize': 25})
        plt.xticks(range(0, 20))
        plt.yticks(range(0, 20))

        bound = np.linspace(0, 1, 4)
        labels = ['Move Left', 'Do Nothing', 'Move Right']
        plt.legend([mpatches.Patch(color=cmap(b)) for b in bound[:-1]], [labels[i] for i in range(3)])
        plt.savefig(f'plots/best_actions_random_{g}_{lr}.png')
        plt.close()

env.close()
