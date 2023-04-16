import gymnasium as gym
import numpy as np
from helper import initialize_grids, initialize_q_table, initialize_state_dict, initialize_random_start, \
    epsilon_greedy_policy, get_closest_in_grid, plot_rewards, plot_steps, evaluate
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches


def train(q, rand_init=True):
    # Initialize variables to track rewards
    reward_list = []
    avg_reward_list = []
    total_steps = []

    for episode in range(n_training_episodes):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        state = env.reset()
        if rand_init:
            state = initialize_random_start(grid_x, grid_v)
        steps = 0
        tot_reward, reward = 0, 0
        terminated = False

        for step in range(max_steps):
            # Choose the action At using epsilon greedy policy
            action = epsilon_greedy_policy(q, state, epsilon, grid_x, grid_v, state_to_qtable, env)

            new_state, reward, terminated, truncated, info = env.step(action)

            s = state_to_qtable[get_closest_in_grid(state, grid_x, grid_v)]
            ns = state_to_qtable[get_closest_in_grid(new_state, grid_x, grid_v)]

            # Calculate current learning rate
            lr = max(min_lr, initial_lr * np.exp(-k * episode))

            # Update Q table
            q[s][action] = q[s][action] + lr * (reward + gamma * np.max(q[ns]) - q[s][action])

            steps += 1

            # If done, finish the episode
            if terminated:  # or truncated:
                total_steps.append(steps)
                break

            # Our state is the new state
            state = new_state

            # Update total reward
            tot_reward += reward

            # Track rewards
            reward_list.append(tot_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(reward_list)
            avg_reward_list.append(avg_reward)
            reward_list = []
            print(f"episode: {episode}\t average reward: {avg_reward}\t avg steps: {np.mean(total_steps[-10:])}"
                  f"\t lr: {lr}\t epsilon: {epsilon}")

    return q, avg_reward_list, total_steps


env = gym.make("MountainCar-v0")

# initialize size of state and action space
state_space = 20 * 20
action_space = env.action_space.n

observation, info = env.reset(seed=42)

# Training parameters
n_training_episodes = 1000  # Total training episodes
initial_lr = 1  # Learning rate       # old: 1
k = 0.005  # lr decay                    # old: 0.005
min_lr = 0.005

# Evaluation parameters
# n_eval_episodes = 100  # Total number of test episodes

# Environment parameters
max_steps = 10000  # Max steps per episode
gamma = 0.95  # Discounting rate
eval_seed = []  # The evaluation seed of the environment

# Exploration parameters
max_epsilon = 0.25   # Exploration probability at start                     # old: 1
min_epsilon = 0.05  # Minimum exploration probability
decay_rate = 0.005  # Exponential decay rate for exploration prob           # old: 0.002


grid_x, grid_v = initialize_grids()
state_to_qtable = initialize_state_dict()

# Training
q_car = initialize_q_table(state_space, action_space)
q_car, avg_rewards, total_steps = train(q_car)

#  Evaluation
n_eval_episodes = 200
trials = 100
rewards = []

for i in range(trials):
    mean_reward, std_reward = evaluate(env, max_steps, n_eval_episodes, q_car, grid_x, grid_v,
                                       state_to_qtable, eval_seed)
    rewards.append(mean_reward)
    if (i+1) % 10 == 0:
        print(f"Trial: {i} - Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

print(f'Average reward over 100 trials: {np.mean(rewards)}')

# Plot V values for random init training
actions = np.max(q_car, axis=1)
actions = actions.reshape((20, 20))
plt.figure(figsize=(16, 12))
sns.heatmap(actions, annot=True)
plt.ylim(0, 20)
plt.xlabel("Position", fontsize=20)
plt.ylabel("Velocity", fontsize=20)
plt.title("V scores for optimal action - random init w/ lr decay", fontdict={'fontsize': 25})
plt.savefig('plots/v_values_rand_decay')
plt.close()

# Plot Q table heatmap for random init
plt.figure(figsize=(12, 12))
cmap = plt.cm.get_cmap('Greens', 3)
plt.pcolor(range(0, 20), range(0, 20), actions, cmap=cmap)
plt.xlabel("Position", fontsize=20)
plt.ylabel("Velocity", fontsize=20)
plt.title("Optimal action at state - random init w/ lr decay", fontdict={'fontsize': 25})
plt.xticks(range(0, 20))
plt.yticks(range(0, 20))

bound = np.linspace(0, 1, 4)
labels = ['Move Left', 'Do Nothing', 'Move Right']
plt.legend([mpatches.Patch(color=cmap(b)) for b in bound[:-1]], [labels[i] for i in range(3)])
plt.savefig('plots/best_actions_rand_decay.png')
plt.close()

# Plot total steps and average rewards for random init
plot_steps(total_steps, 'rand_decay')
plot_rewards(avg_rewards, 'rand_decay')

# Save Q table
np.savetxt('data/q_rand_decay.txt', q_car)
np.array(avg_rewards)
np.savetxt('data/avg_rewards_rand_decay.txt', avg_rewards)
np.array(total_steps)
np.savetxt('data/total_steps_rand_decay.txt', total_steps)

env.close()
