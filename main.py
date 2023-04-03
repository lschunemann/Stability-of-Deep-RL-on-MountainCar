import gymnasium as gym
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from helper import initialize_grids, initialize_q_table, initialize_state_dict, epsilon_greedy_policy, greedy_policy,\
    get_closest_in_grid

env = gym.make("MountainCar-v0")

# initialize size of state and action space
state_space = 20 * 20
action_space = env.action_space.n

observation, info = env.reset(seed=42)


def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, q):
    # Initialize variables to track rewards
    reward_list = []
    avg_reward_list = []
    total_steps = []

    for episode in range(n_training_episodes):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        state = env.reset()
        step = 0
        tot_reward, reward = 0, 0
        terminated = False

        for step in range(max_steps):
            # Choose the action At using epsilon greedy policy
            action = epsilon_greedy_policy(q, state, epsilon, grid_x, grid_v, state_to_qtable, env)

            new_state, reward, terminated, truncated, info = env.step(action)

            s = state_to_qtable[get_closest_in_grid(state, grid_x, grid_v)]
            ns = state_to_qtable[get_closest_in_grid(new_state, grid_x, grid_v)]

            # Update Q table
            q[s][action] = q[s][action] + learning_rate * (reward + gamma * np.max(q[ns]) - q[s][action])

            step += 1

            # If done, finish the episode
            if terminated:  # or truncated:
                total_steps.append(step)
                break

            # Our state is the new state
            state = new_state

            # Update total reward
            tot_reward += reward

            # Track rewards
            reward_list.append(tot_reward)

        if (episode+1) % 10 == 0:
            avg_reward = np.mean(reward_list)
            avg_reward_list.append(avg_reward)
            reward_list = []
            print(f"episode: {episode}\t average reward: {avg_reward}\t avg steps: {np.mean(total_steps[-10:])}")

    return q, avg_reward_list, total_steps


def evaluate_agent(env, max_steps, n_eval_episodes, q, seed):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param q: The Q-table
    :param seed: The evaluation seed array (for taxi-v3)
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        if seed:
            state = env.reset(seed=seed[episode])
        else:
            state = env.reset()
        step = 0
        terminated = False
        total_rewards_ep = 0

        for step in range(max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            # s = state_to_qtable[get_closest_in_grid(state, grid_x, grid_v)]
            # action = np.argmax(q[s][:])
            action = greedy_policy(q, state, grid_x, grid_v, state_to_qtable)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward

            if terminated:  # or truncated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


# Training parameters
n_training_episodes = 1000  # Total training episodes
learning_rate = 0.1  # Learning rate

# Evaluation parameters
n_eval_episodes = 100  # Total number of test episodes

# Environment parameters
max_steps = int(100000)  # Max steps per episode
gamma = 0.95  # Discounting rate
eval_seed = []  # The evaluation seed of the environment

# Exploration parameters
max_epsilon = 1.0   # Exploration probability at start
min_epsilon = 0.05  # Minimum exploration probability
decay_rate = 0.002  # Exponential decay rate for exploration prob


grid_x, grid_v = initialize_grids()

state_to_qtable = initialize_state_dict()

q_car = initialize_q_table(state_space, action_space)
q_car, avg_rewards, total_steps = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, q_car)

mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, q_car, eval_seed)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

# Plot Heatmap of Q Table
fig = plt.figure()
ax = sns.heatmap(q_car)
fig.savefig('q_heatmap.png')
plt.close(fig)

# Plot Rewards
plt.plot(1*(np.arange(len(avg_rewards)) + 1), avg_rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward per Episode')
plt.savefig('rewards.png')
plt.close()

# Plot Steps
plt.plot(1*(np.arange(len(total_steps)) + 1), total_steps)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode')
plt.savefig('steps.png')
plt.close()

# Plot Q table heatmap
chosen = np.max(q_car, axis=1)
chosen = chosen.reshape((20, 20))
plt.figure(figsize=(12, 12))
cmap = plt.cm.get_cmap('Greens', 8)
plt.pcolor(range(0, 20), range(0, 20), chosen, cmap=cmap)
plt.xlabel("Position", fontsize=15)
plt.ylabel("Velocity", fontsize=15)
plt.savefig('best_actions.png')
plt.close()

# Save Q table
np.savetxt('q_car.txt', q_car)

env.close()
