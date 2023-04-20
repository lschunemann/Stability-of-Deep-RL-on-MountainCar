import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import gymnasium as gym
import numpy as np
import random

device = torch.device("cpu")

# Hyperparameters
n_training_episodes = 200
gamma = 0.99
learning_rate = 0.1

# Exploration parameters
epsilon_max = 0.5
epsilon_min = 0.05
decay_rate = 0.005


class TrainMountainCar:
    def __init__(self, n_training_episodes=200, gamma=0.99, learning_rate=0.1, epsilon_max=0.5,
                 epsilon_min=0.05, decay_rate=0.005, max_steps=10000):
        self.n_training_episodes = n_training_episodes
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        self.max_steps = max_steps

    def epsilon_greedy_policy(self, state, epsilon, env):
        if random.uniform(0, 1) < epsilon:
            return env.action_space.sample()
        else:
            return None  # TODO

    def train(self, debug=False):
        env = gym.make('MountainCar-v0')

        total_steps = []
        total_rewards = []

        for episode in range(self.n_training_episodes):
            steps = 0
            rewards = 0

            state = env.reset()
            epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-decay_rate * episode)

            for step in range(self.max_steps):
                # Choose the action At using epsilon greedy policy
                action = self.epsilon_greedy_policy(state, epsilon, env)

                new_state, reward, terminated, truncated, info = env.step(action)

                steps += 1

                # If done, finish the episode
                if terminated:  # or truncated:
                    total_steps.append(steps)
                    break

                state = new_state

                # Update total reward
                rewards += reward

                # Track rewards
                total_rewards.append(rewards)

                if debug:
                    if (episode + 1) % 10 == 0:
                        print(f"episode: {episode}\t avg steps: {np.mean(total_steps[-10:])}")\

        return total_rewards, total_steps
