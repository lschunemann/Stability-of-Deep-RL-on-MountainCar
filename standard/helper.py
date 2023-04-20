import random
import numpy as np
import matplotlib.pyplot as plt


def initialize_grids():
    # initialize bins for x-axis and velocity
    grid_x = []
    grid_v = []
    for i in range(20):
        grid_x.append(round(-1.2 + i * 0.1, 2))
        grid_v.append(round(-0.07 + i * 0.008, 3))  # 0.008 0.00735
    return grid_x, grid_v


def initialize_state_dict():
    state_to_qtable = {}
    x = -1.2
    v = -0.07
    i = 0
    while x <= 0.6:
        state_to_qtable[(x, v)] = i
        i += 1
        while v <= 0.074:
            state_to_qtable[(x, v)] = i
            v += 0.008  # 0.008 0.00735
            v = round(v, 3)  # rounding to avoid floating point errors
            i += 1
        v = -0.07
        x += 0.1
        x = round(x, 2)  # rounding to avoid floating point errors
    return state_to_qtable


def get_closest_in_grid(n: np.ndarray, grid_x: list, grid_v: list) -> tuple:
    """round tuple n to the closest representation within the grid"""
    if type(n) != tuple:
        x = n[0]
        v = n[1]
    else:
        x = n[0][0]
        v = n[0][1]

    x = min(grid_x, key=lambda a: abs(a-x))
    v = min(grid_v, key=lambda a: abs(a-v))

    return x, v


def initialize_q_table(state_space, action_space):
    q = np.zeros((state_space, action_space))
    return q


def epsilon_greedy_policy(q, state, epsilon, grid_x, grid_v, state_to_qtable, env):
    if random.uniform(0, 1) > epsilon:
        s = state_to_qtable[get_closest_in_grid(state, grid_x, grid_v)]
        choice = np.argmax(q[s])
    else:
        choice = env.action_space.sample()  # Take a random action
    return choice


def greedy_policy(q, state, grid_x, grid_v, state_to_qtable):
    s = state_to_qtable[get_closest_in_grid(state, grid_x, grid_v)]
    choice = np.argmax(q[s])
    return choice


def initialize_random_start(grid_x, grid_v):
    x = round(random.uniform(-1.2, 0.6), 2)
    v = round(random.uniform(-0.07, 0.07), 3)
    x = min(grid_x, key=lambda a: abs(a-x))
    v = min(grid_v, key=lambda a: abs(a-v))
    return np.array([x, v])


def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, q, grid_x, grid_v, learning_rate,
          state_to_qtable, gamma, rand_init=False):
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

            # Update Q table
            q[s][action] = q[s][action] + learning_rate * (reward + gamma * np.max(q[ns]) - q[s][action])

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
            print(f"episode: {episode}\t average reward: {avg_reward}\t avg steps: {np.mean(total_steps[-10:])}")

    return q, avg_reward_list, total_steps


def evaluate(env, max_steps, n_eval_episodes, q, grid_x, grid_v, state_to_qtable, seed):
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


def evaluate_with_steps(env, max_steps, n_eval_episodes, q, grid_x, grid_v, state_to_qtable, seed):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param q: The Q-table
    :param seed: The evaluation seed array (for taxi-v3)
    """
    episode_rewards = []
    total_steps = []
    for episode in range(n_eval_episodes):
        if seed:
            state = env.reset(seed=seed[episode])
        else:
            state = env.reset()
        steps = 0
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
            steps += 1
        episode_rewards.append(total_rewards_ep)
        total_steps.append(steps)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_steps = np.mean(total_steps)

    return mean_reward, std_reward, mean_steps


def plot_rewards(avg_rewards, title):
    # Plot Rewards
    plt.plot(10 * (np.arange(len(avg_rewards)) + 1), avg_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title(f'Average Reward per Episode - {title}')
    plt.savefig(f'plots/rewards_{title}.png')
    plt.close()


def plot_steps(total_steps, title):
    # Plot Steps
    plt.plot(np.arange(len(total_steps)) + 1, total_steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title(f'Steps per Episode - {title}')
    plt.savefig(f'plots/steps_{title}.png')
    plt.close()
