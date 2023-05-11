import torch
import torchvision.transforms as transforms
import numpy as np
from models import DQN_square
import gymnasium as gym
import random
from helper_DQN import scale_and_resize
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device("cuda")
transform = scale_and_resize()

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class TrainMountainCar:
    def __init__(self, n_training_episodes=200, gamma=0.99, learning_rate=0.1, epsilon_max=0.5,
                 epsilon_min=0.05, decay_rate=0.005, max_steps=10000, batch_size=32, debug=False):
        self.n_training_episodes = n_training_episodes
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.eval_every = 50
        self.eval_episodes = 50
        self.eval_epsilon = 0.5
        self.debug = debug

    def epsilon_greedy_policy(self, policy: torch.nn.Module, X, epsilon: float, env: gym.envs):
        """
        Samples a random action with probability epsilon and picks the maximum action under policy network otherwise.
        :param policy: Policy Network under which to take action
        :param X: stacked tensor of shape (4,80,120)
        :param epsilon: float probability of sampling a random action
        :param env: Gymnasium environment
        :return: Randomly sampled action or maximum action under policy network
        """
        if random.uniform(0, 1) < epsilon:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                X = X.unsqueeze(0).to(device)
                return policy(X).max(1)[1].view(1, 1).item()

    def prepare_images(self, env, action):
        """
        performs the action in the environment 4 consecutive times and gets the images after each step, preprocesses
        them and returns them as a list.
        :param env: Gymnasium environment
        :param action: Action to be performed
        :return: list of stacked images, rewards after the 4 steps, whether one of the experienced states was terminal
        """
        stacked_images = []
        terminated = False
        for i in range(4):
            _, _, stop, _, _ = env.step(action)
            terminated = stop if stop else terminated       # if one of the visited states is terminal, return terminal
            img = env.render()
            img = transforms.ToTensor()(img)
            stacked_images.append(torch.squeeze(transform(img)))
        reward = 0 if terminated else -1
        return stacked_images, reward, terminated

    def initialize_measuring_states(self, env):
        """
        Randomly samples 200 states by taking random actions
        :param env: Gymnasium environment
        :return: list of states that were visited by random walk
        """
        measuring_states = []
        env.reset()
        for i in range(200):
            action = env.action_space.sample()
            stacked_img, _, _ = self.prepare_images(env, action)
            measuring_states.append(stacked_img)
        env.reset()
        return measuring_states

    def eval(self, policy: torch.nn.Module, env: gym.envs):
        """
        Evaluate a policy and return the average reward over self.eval_episodes trials with maximum 10000 steps each
        :param policy: The policy to be evaluated
        :param env: The Gymnasium environment
        :return: average over rewards collected turing trials
        """
        rewards_list = []
        for episode in range(self.eval_episodes):
            env.reset()

            # min 4 and up to 30 no-op actions (at least 4 to have stacked input to start training on)
            noop = random.randint(0, 26)
            for i in range(noop):
                action = env.action_space.sample()
                env.step(action)

            stacked_images, _, _ = self.prepare_images(env, env.action_space.sample())
            X = torch.stack(stacked_images)
            rewards = 0

            for i in range(self.max_steps):      # max episode length 10000
                action = self.epsilon_greedy_policy(policy, X, self.eval_epsilon, env)
                stacked_images, reward, terminated = self.prepare_images(env, action)

                # update image stack with new state
                X = torch.stack(stacked_images)

                rewards += reward

                if terminated:
                    break

            rewards_list.append(rewards)

        return np.mean(rewards_list)

    def train(self):
        """
        trains DQN using a fixed target network if self.fixed_target == True, otherwise with the policy network.
        :return: list of total rewards, list of steps in each episode, q values over sampled states
        """
        env = gym.make('MountainCar-v0', render_mode='rgb_array')

        # keep track of total steps and rewards
        total_steps = 0
        total_rewards = []
        total_steps_list = []
        evaluations = []

        # initialize states in which Q value is measured every X episodes to track progress
        measuring_states = self.initialize_measuring_states(env)
        q_measures = []

        policy = DQN_square(env.action_space.n).to(device)

        # Best values found during evaluation
        best_reward = - float('inf')
        best_policy = policy.state_dict()

        optimizer = torch.optim.RMSprop(policy.parameters(), lr=self.learning_rate, weight_decay=0.99, momentum=0.95)      # remove momentum??

        for episode in range(self.n_training_episodes):
            steps = 0
            # total_steps += 4
            rewards = 0

            env.reset()

            # min 4 and up to 30 no-op actions (at least 4 to have stacked input to start training on)
            noop = random.randint(0, 26)
            for i in range(noop):
                action = env.action_space.sample()
                env.step(action)

            # pick a random action at the start and observe four frames
            action = env.action_space.sample()
            stacked_images, _, _ = self.prepare_images(env, action)

            # Input tensor
            X = torch.stack(stacked_images).unsqueeze(0).to(device)

            while True:
                # linear epsilon decay based on steps
                epsilon = max(self.epsilon_max - ((self.epsilon_max - self.epsilon_min)/1000000) * total_steps, self.epsilon_min)

                # Choose the action At using epsilon greedy policy
                action = self.epsilon_greedy_policy(policy, X, epsilon, env)
                # take action
                stacked_images, reward, terminated = self.prepare_images(env, action)
                reward = torch.tensor([reward])

                # update image stack with new state
                X_new = torch.stack(stacked_images).unsqueeze(0).to(device)

                steps += 4
                total_steps += 4

                state_action_values = policy(X).gather(1, action.type(torch.int64))
                # update network
                next_state_values = policy(X_new).max(1)[0].detach()
                # Compute the expected Q values
                expected_state_action_values = (next_state_values * self.gamma) + reward

                # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
                loss = torch.nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1)).squeeze()

                # loss = loss.mean()

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.)    # clip gradients
                optimizer.step()

                # Update total reward
                rewards += reward

                # update current state to be next state
                X = X_new

                # If done, finish the episode
                if terminated or steps >= self.max_steps:  # or truncated:
                    # Track rewards
                    total_rewards.append(rewards)
                    total_steps_list.append(steps)

                    # measure Q values in selected states
                    Z = [torch.stack(measuring_state) for measuring_state in measuring_states]
                    Q_states = torch.stack(Z).to(device)
                    Q_states = torch.unique(Q_states, dim=0, sorted=False)  # eliminate duplicate states
                    with torch.no_grad():
                        q_measures.append(torch.mean(policy(Q_states).max(1)[0]).item())

                    # Evaluate current policy and save optimal policy weights
                    if episode > 0 and episode % self.eval_every == 0:
                        eval_reward = self.eval(policy, env)
                        if eval_reward > best_reward:
                            best_reward = eval_reward
                            best_policy = policy.state_dict()
                        print(f"Evaluation: {int(episode/self.eval_every)}\t average reward: {eval_reward}")
                        evaluations.append(eval_reward)

                    # print training information
                    if self.debug:
                        print(f"episode: {episode + 1:03d}\t steps: {steps + 1:05d}\t total steps:"
                              f"{total_steps + 1:06d}\t epsilon: {epsilon:.2f}\t average Q: {q_measures[-1]:.3f}")
                    break

        return total_rewards, total_steps_list, q_measures, best_policy, evaluations


# Hyperparameters
n_training_episodes = 500
gamma = 0.99
learning_rate = 0.00025
max_training_steps = 10000

# Exploration parameters
epsilon_max = 1
epsilon_min = 0.1

# replay memory parameters
batch_size = 32

car = TrainMountainCar(n_training_episodes=n_training_episodes, gamma=gamma, learning_rate=learning_rate,
                       epsilon_max=epsilon_max, epsilon_min=epsilon_min,
                       max_steps=max_training_steps, batch_size=batch_size)

total_rewards, total_steps_list, q_measures, best_policy, evaluations = car.train()

# save best policy as well as steps and q measures
torch.save(best_policy, 'data/DQN_no_exp_replay.pth')
np.savetxt(f'data/steps_DQN_no_exp_replay.txt', total_steps_list)
np.savetxt(f'data/q_values_DQN_no_exp_replay.txt', q_measures)
np.savetxt(f'data/eval_DQN_no_exp_replay.txt', evaluations)


# Plot steps per episode
plt.plot(np.arange(len(total_steps_list)) + 1, total_steps_list)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode - DQN_no_exp_replay')
plt.savefig('plots/steps_DQN_no_exp_replay.png')
plt.close()

# Plot q measures per episode
plt.plot(np.arange(len(q_measures)) + 1, q_measures)
plt.xlabel('Episode')
plt.ylabel('Average Q')
plt.title('Average Q measure over sampled states')
plt.savefig('plots/q_measures_DQN_no_exp_replay.png')
plt.close()
