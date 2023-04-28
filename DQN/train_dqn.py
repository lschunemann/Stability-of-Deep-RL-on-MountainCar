import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import random
from helper_DQN import transform_to_grayscale_and_normalize, ReplayMemory
from models import DQN

device = torch.device("cuda")

# Hyperparameters
n_training_episodes = 2
gamma = 0.99
learning_rate = 0.1
max_training_steps = 10000

# Exploration parameters
epsilon_max = 0.9
epsilon_min = 0.1
decay_rate = 0.0005

# replay memory parameters
replay_size = 10000
batch_size = 32


class TrainMountainCar:
    def __init__(self, n_training_episodes=200, gamma=0.99, learning_rate=0.1, epsilon_max=0.5,
                 epsilon_min=0.05, decay_rate=0.005, max_steps=10000, batch_size=32):
        self.n_training_episodes = n_training_episodes
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        self.max_steps = max_steps
        self.batch_size = batch_size

    def epsilon_greedy_policy(self, policy, X, epsilon, env):
        if random.uniform(0, 1) < epsilon:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                X = X.unsqueeze(0).to(device)
                return policy(X).max(1)[1].view(1, 1).item()

    def prepare_images(self, env):
        stacked_images = []
        img = env.render()
        stacked_images.append(transform_to_grayscale_and_normalize(img))

        for i in range(3):
            _, _, _, _, _ = env.step(env.action_space.sample())
            img = env.render()
            stacked_images.append(transform_to_grayscale_and_normalize(img))
        return stacked_images

    def update_img_stack(self, env, images):
        img = env.render()
        images = images[1:]
        images.append(transform_to_grayscale_and_normalize(img))
        X_new = torch.stack(images)
        return images, X_new

    def train(self, debug=False):
        env = gym.make('MountainCar-v0', render_mode='rgb_array')

        # keep track of total steps and rewards (start total steps as -1 because of initial frame stack)
        total_steps = -1
        total_rewards = []

        experience_memory = ReplayMemory(replay_size)

        policy = DQN(env.action_space.n).to(device)

        optimizer = torch.optim.RMSprop(policy.parameters(), lr=self.learning_rate)

        for episode in range(self.n_training_episodes):
            steps = 3
            total_steps += 4
            rewards = 0

            env.reset()

            # stack 4 images transformed to grayscale
            stacked_images = self.prepare_images(env)

            for step in range(self.max_steps):
                # epsilon decay
                epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-decay_rate * total_steps)

                # Input tensor
                X = torch.stack(stacked_images)

                # Choose the action At using epsilon greedy policy
                action = self.epsilon_greedy_policy(policy, X, epsilon, env)
                # take action
                _, reward, terminated, truncated, _ = env.step(action)
                reward = torch.tensor([reward])

                # update image stack with new state
                stacked_images, X_new = self.update_img_stack(env, stacked_images)

                # update experience memory
                experience_memory.add([X, action, reward, X_new], terminated)

                steps += 1
                total_steps += 1

                if len(experience_memory) > self.batch_size:
                    experiences = experience_memory.sample(self.batch_size)
                    experiences = np.asarray(experiences, dtype=object)
                    experience = experiences[:, 0]
                    experience = np.asarray([np.array(i, dtype=object) for i in experience])
                    states = experience[:, 0]
                    a = experience[:, 1]
                    r = experience[:, 2]
                    next_states = experience[:, 3]
                    terminations = experiences[:, 1]
                    mask = [i for i, x in enumerate(terminations) if x]

                    # change states and rewards back to tensors
                    states = np.vstack(states).astype(np.float32)
                    states = torch.from_numpy(states)
                    next_states = np.vstack(next_states).astype(np.float32)
                    next_states = torch.from_numpy(next_states)
                    r = np.vstack(r).astype(np.float32)
                    r = torch.from_numpy(r).to(device)
                    a = np.vstack(a).astype(np.float32)
                    a = torch.from_numpy(a).to(device)
                    states = torch.reshape(states, (32, 4, 400, 600)).to(device)
                    next_states = torch.reshape(next_states, (32, 4, 400, 600)).to(device)

                    # calculate target
                    state_action_values = policy(states).gather(1, a.type(torch.int64))
                    # target = r + self.gamma * np.argmax(values(x_new))
                    # update network
                    next_state_values = torch.zeros(self.batch_size, device=device)
                    next_state_values[mask] = policy(next_states[mask]).max(1)[0].detach()
                    # Compute the expected Q values
                    expected_state_action_values = (next_state_values * self.gamma) + r.squeeze(1)

                    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
                    # Optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    # for param in policy.parameters():
                    #     param.grad.data.clamp_(-1, 1)
                    optimizer.step()

                # Update total reward
                rewards += reward

                print(f"episode: {episode}\t steps: {steps}\t total steps: {total_steps}\t epsilon: {epsilon:.2f}")

                # if debug:
                #     if (episode + 1) % 10 == 0:
                #         print(f"episode: {episode}\t avg rewards: {np.mean(total_rewards[-10:]), steps: {steps}}")

                # If done, finish the episode
                if terminated: # or truncated:
                    # Track rewards
                    total_rewards.append(rewards)
                    break

        torch.save(policy.state_dict(), 'data/DQN.pth')

        return total_rewards, total_steps


car = TrainMountainCar(n_training_episodes=n_training_episodes, gamma=gamma, learning_rate=learning_rate,
                       epsilon_max=epsilon_max, epsilon_min=epsilon_min, decay_rate=decay_rate,
                       max_steps=max_training_steps, batch_size=batch_size)

car.train(True)

