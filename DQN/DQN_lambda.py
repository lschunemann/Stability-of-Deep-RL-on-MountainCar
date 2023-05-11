import numpy as np
import torch
from lambda_memory import make_replay_memory
import gymnasium as gym
from models import MLP_state
import random


def calculate_lambda_returns(rewards, qvalues, dones, mask, discount, lambd):
    dones = dones.astype(np.float32)
    qvalues[-1] *= (1.0 - dones[-1])
    lambda_returns = rewards + (discount * qvalues[1:])
    for i in reversed(range(len(rewards) - 1)):
        a = lambda_returns[i] + (discount * lambd * mask[i]) * (lambda_returns[i+1] - qvalues[i+1])
        b = rewards[i]
        lambda_returns[i] = (1.0 - dones[i]) * a + dones[i] * b
    return lambda_returns


def epsilon_greedy_policy(policy: torch.nn.Module, X, epsilon: float, env: gym.envs):

    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            # X = X.unsqueeze(0).to(device)
            # X = torch.tensor(X, device=device, dtype=torch.float)
            return policy(X).max(1)[1].view(1, 1).item()


def train():
    total_steps_list = []
    total_steps = 0
    for episode in range(n_training_episodes):
        steps = 0
        t = episode

        obs = env.reset()[0]
        # noop = random.randint(0, 30)
        # for i in range(noop):
        #     action = env.action_space.sample()
        #     obs = env.step(action)

        for step in range(max_training_steps):
            # linear epsilon decay based on steps
            epsilon = max(epsilon_max - ((epsilon_max - epsilon_min) / 500000) * total_steps,
                          epsilon_min)

            # Choose the action At using epsilon greedy policy
            replay_memory.store_obs(obs)
            state = replay_memory.encode_recent_observation()

            X = torch.tensor(state, device=device, dtype=torch.float)
            action = epsilon_greedy_policy(policy, X, epsilon, env)
            obs, reward, done, _, _ = env.step(action)
            reward = torch.tensor([reward])
            replay_memory.store_effect(action, reward, done)

            steps += 1
            total_steps += 1

            t = step

            t -= prepopulate  # Make relative to training start
            if t >= 0:
                train_frac = max(0.0, (t - prepopulate) / (max_training_steps - prepopulate))
                replay_memory.refresh(train_frac)
                # extract old states, actions, rewards and new states from ReplayMemory
                experiences = replay_memory.sample(batch_size)
                states, actions, _rewards, next_states, terminations = (i for i in zip(*experiences))

                a = (torch.tensor(actions).long().unsqueeze(dim=1)).to(device)
                r = torch.tensor(_rewards).unsqueeze(dim=1).to(device)
                states = np.vstack(states).astype(np.float32)
                states = torch.from_numpy(states)
                next_states = np.vstack(next_states).astype(np.float32)
                next_states = torch.from_numpy(next_states)
                states = torch.reshape(states, (32, 4, 84, 84)).to(device)  # 80,120
                next_states = torch.reshape(next_states, (32, 4, 84, 84)).to(device)
                mask = [i for i, x in enumerate(terminations) if not x]  # get all non-final states

                state_action_values = policy(states).gather(1, a.type(torch.int64))
                # target = r + self.gamma * np.argmax(values(x_new))
                # update network
                next_state_values = torch.zeros(batch_size, device=device)

                next_state_values[mask] = policy(next_states[mask]).max(1)[0].detach()
                # Compute the expected Q values
                expected_state_action_values = (next_state_values * 0.99) + r.squeeze(1)

                loss = torch.nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1)).squeeze()

                loss = loss.mean()

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.)  # clip gradients
                optimizer.step()

            if done:
                total_steps_list.append(steps)
                print(f"episode: {episode + 1:03d}\t steps: {steps + 1:05d}\t total steps:"
                      f"{total_steps + 1:06d}\t epsilon: {epsilon:.2f}")

    return total_steps_list


device = torch.device("cuda")

# Hyperparameters
n_training_episodes = 1000
max_training_steps = 10000

# Exploration parameters
epsilon_max = 1
epsilon_min = 0.1

batch_size = 32

prepopulate = 1000

env = gym.make("MountainCar-v0")

policy = MLP_state(env.action_space.n).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

replay_memory = make_replay_memory(capacity=500000, history_len=1, discount=0.99,
                                   cache_size=80000, block_size=100, priority=0.0, lambd=0.9)

total_steps_list = train()

env.close()
