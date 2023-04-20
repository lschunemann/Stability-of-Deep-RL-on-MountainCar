import gymnasium as gym
import numpy as np
from helper import initialize_grids, initialize_q_table, initialize_state_dict, train, evaluate

env = gym.make("MountainCar-v0")

# initialize size of state and action space
state_space = 20 * 20
action_space = env.action_space.n

observation, info = env.reset(seed=42)


# Training parameters
n_training_episodes = 1000  # Total training episodes
learning_rate = 0.1  # Learning rate

# Evaluation parameters
n_eval_episodes = 100  # Total number of test episodes

# Environment parameters
max_steps = int(10000)  # Max steps per episode
gamma = 0.95  # Discounting rate
eval_seed = []  # The evaluation seed of the environment

# Exploration parameters
max_epsilon = 1.0   # Exploration probability at start
min_epsilon = 0.05  # Minimum exploration probability
decay_rate = 0.002  # Exponential decay rate for exploration prob


grid_x, grid_v = initialize_grids()

state_to_qtable = initialize_state_dict()

q_car = initialize_q_table(state_space, action_space)
q_car, avg_rewards, total_steps = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps,
                                        q_car, grid_x, grid_v, learning_rate, state_to_qtable, gamma, False)

mean_reward, std_reward = evaluate(env, max_steps, n_eval_episodes, q_car, grid_x, grid_v, state_to_qtable, eval_seed)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")


# Save Q table
np.savetxt('data/q_car.txt', q_car)
np.array(avg_rewards)
np.savetxt('data/avg_rewards.txt', avg_rewards)
np.array(total_steps)
np.savetxt('data/total_steps.txt', total_steps)

env.close()
