import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from helper import plot_steps, plot_rewards

q = np.loadtxt('data/q_car.txt')
total_steps_main = np.loadtxt('data/total_steps_main.txt')
avg_reward_main = np.loadtxt('data/avg_reward_main.txt')

# Plot Q table heatmap
chosen = np.max(q, axis=1)
chosen = chosen.reshape((20, 20))
plt.figure(figsize=(12, 12))
cmap = plt.cm.get_cmap('Greens', 3)
plt.pcolor(range(0, 20), range(0, 20), chosen, cmap=cmap)
plt.xlabel("Position", fontsize=20)
plt.ylabel("Velocity", fontsize=20)
plt.title("Optimal action at state", fontdict={'fontsize': 25})
plt.xticks(range(0, 20))
plt.yticks(range(0, 20))

bound = np.linspace(0, 1, 4)
labels = ['Move Left', 'Do Nothing', 'Move Right']
plt.legend([mpatches.Patch(color=cmap(b)) for b in bound[:-1]], [labels[i] for i in range(3)])
plt.savefig('plots/best_actions.png')
plt.close()

# Plot V values over state space
plt.figure(figsize=(16, 12))
sns.heatmap(chosen, annot=True)
plt.ylim(0, 20)
plt.xlabel("Position", fontsize=20)
plt.ylabel("Velocity", fontsize=20)
plt.title("V scores for optimal action", fontdict={'fontsize': 25})
plt.savefig('plots/v_values')

# Plot V values for each action over state space
reshaped = np.reshape(q, (20, 20, 3))

for i in range(3):
    plt.figure(figsize=(16, 12))
    sns.heatmap(reshaped[:, :, i], annot=True)
    plt.ylim(0, 20)
    plt.xlabel("Position", fontsize=20)
    plt.ylabel("Velocity", fontsize=20)
    plt.title(f"V scores for action: {labels[i]}", fontdict={'fontsize': 25})
    plt.savefig(f'plots/v_values_{i}')

# Plot Total Steps and average rewards
plot_rewards(avg_reward_main, 'main')
plot_steps(total_steps_main, 'main')


# Plot V values for random init training
q_rand = np.loadtxt('data/q_random.txt')
actions = np.max(q_rand, axis=1)
actions = actions.reshape((20, 20))
plt.figure(figsize=(16, 12))
sns.heatmap(actions, annot=True)
plt.ylim(0, 20)
plt.xlabel("Position", fontsize=20)
plt.ylabel("Velocity", fontsize=20)
plt.title("V scores for optimal action - random init", fontdict={'fontsize': 25})
plt.savefig('plots/v_values_random')

# Plot Q table heatmap for random init
plt.figure(figsize=(12, 12))
cmap = plt.cm.get_cmap('Greens', 3)
plt.pcolor(range(0, 20), range(0, 20), actions, cmap=cmap)
plt.xlabel("Position", fontsize=20)
plt.ylabel("Velocity", fontsize=20)
plt.title("Optimal action at state - random init", fontdict={'fontsize': 25})
plt.xticks(range(0, 20))
plt.yticks(range(0, 20))

bound = np.linspace(0, 1, 4)
labels = ['Move Left', 'Do Nothing', 'Move Right']
plt.legend([mpatches.Patch(color=cmap(b)) for b in bound[:-1]], [labels[i] for i in range(3)])
plt.savefig('plots/best_actions_random.png')
plt.close()

# Plot total steps and average rewards for random init
total_steps_rand = np.loadtxt('data/total_steps_rand.txt')
avg_reward_rand = np.loadtxt('data/avg_steps_rand.txt')
plot_steps(total_steps_rand, 'rand')
plot_rewards(avg_reward_rand, 'rand')
