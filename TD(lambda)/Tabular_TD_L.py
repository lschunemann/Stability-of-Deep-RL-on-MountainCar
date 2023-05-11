import random
from standard.helper import initialize_grids, initialize_q_table, initialize_state_dict, get_closest_in_grid
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# class TabularTDLambda:
#     def __init__(self, env, lmbda=0.5, gamma=0.99, lr=0.1):
#         self.lmbda = lmbda
#         self.gamma = gamma
#         self.lr = lr
#         self.values = np.zeros(env.observation_space.n, np.float64)
#         self.eligibility_trace = np.zeros_like(self.values, np.float64)
#
#     def get_action(self, env, epsilon):
#         if random.uniform(0, 1) < epsilon:
#             return env.action_space.sample()
#
#     def update(self, env, state, reward, next_state):
#         delta = reward + self.gamma * self.values[next_state] - self.values[state]
#         for x in range(env.observation_space.n):
#             self.eligibility_trace[x] = self.gamma * self.lmbda * self.eligibility_trace[x]
#             if state == x:
#                 self.eligibility_trace[x] = 1
#             self.values[x] += self.lr * delta * self.eligibility_trace[x]

def epsilon_greedy(env, state, epsilon, state_to_qtable, q):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    s = state_to_qtable[get_closest_in_grid(state, grid_x, grid_v)]
    return np.argmax(q[s])


def train(q, env, epsilon, num_training_episodes, gamma, lmbda, grid_x, grid_v, lr, debug=True):

    steps_list = []

    for episode in range(num_training_episodes):
        state = env.reset()[0]
        z = np.zeros(q.shape)
        steps = 0

        while True:
            action = epsilon_greedy(env, state, epsilon, state_to_qtable, q)

            new_state, reward, terminated, truncated, info = env.step(action)

            # Update eligibility trace
            z = gamma * lmbda * z
            z[state_to_qtable[get_closest_in_grid(state, grid_x, grid_v)]][action] += 1

            delta = reward + gamma * (q[state_to_qtable[get_closest_in_grid(new_state, grid_x, grid_v)]]
                                      - q[state_to_qtable[get_closest_in_grid(state, grid_x, grid_v)]])

            q = q + lr * delta * z

            # If done, finish the episode
            if terminated:
                steps_list.append(steps)
                if debug and (episode+1) % 10 == 0:
                    print(f"Training episode: {episode}, lr: {lr}, lambda: {lmbda} - steps: {steps}")
                break

            steps += 1
            state = new_state

    return steps_list


env = gym.make("MountainCar-v0")

# initialize size of state and action space
state_space = 20 * 20
action_space = env.action_space.n

# initialize discretization and mapping from state to q table
grid_x, grid_v = initialize_grids()
state_to_qtable = initialize_state_dict()

# Training hyperparameters
epsilon = 0.05
num_training_episodes = 500
gamma = 0.99
lmbda = 0.5
lr = 0.1

# Training
q_car = initialize_q_table(state_space, action_space)
q_car, avg_rewards, total_steps = train(q_car, env, epsilon, num_training_episodes, gamma, lmbda, grid_x, grid_v, lr)


actions = np.max(q_car, axis=1)
actions = actions.reshape((20, 20))
plt.figure(figsize=(16, 12))
ax = plt.subplot(111)
ax = sns.heatmap(actions, annot=True)
plt.ylim(0, 20)
plt.xlabel("Position", fontsize=20)
plt.ylabel("Velocity", fontsize=20)
plt.title("Q values for optimal action - Tabular TD lambda", fontdict={'fontsize': 25})
plt.savefig('plots/q_values_Tabular_TD_L')

# Plot Steps
plt.plot(np.arange(len(total_steps)) + 1, total_steps)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode - Tabular TD lambda')
plt.savefig('plots/steps_Tabular_TD_L.png')
plt.close()

np.savetxt("q_TD_L", q_car)
