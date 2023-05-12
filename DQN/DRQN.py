import torch
import numpy as np
import matplotlib.pyplot as plt
from helper_DQN import running_mean, scale_and_resize, RecurrentExperienceMemory
import collections
import random
import gymnasium as gym
from models import DRQN
import torchvision.transforms as transforms

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device("cuda")


class TrainRecurrentMountainCar:
    def __init__(self, n_training_episodes=200, gamma=0.99, learning_rate=0.1, epsilon_max=0.5,
                 epsilon_min=0.05, max_steps=10000, batch_size=32, fixed_target=False,
                 copy_target=10000, replay_size=100000, double=False, dueling=False, prioritized=False, debug=False,
                 eval_epsilon=0.05, eval_episodes=25, eval_every=50, noisy=False, distributional=False, env=None,
                 epsilon_frame=500000):
        self.n_training_episodes = n_training_episodes
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_frame = epsilon_frame
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.fixed_target = fixed_target
        self.copy_target = copy_target
        self.replay_size = replay_size
        self.double = double
        self.dueling = dueling
        self.debug = debug
        self.eval_epsilon = eval_epsilon
        self.eval_episodes = eval_episodes
        self.eval_every = eval_every
        self.sequence_length = 10
        self.seq = [np.zeros(1) for _ in range(self.sequence_length)]

    # def epsilon_greedy_policy(self, policy: torch.nn.Module, X, epsilon: float, env: gym.envs):
    #     """
    #     Samples a random action with probability epsilon and picks the maximum action under policy network otherwise.
    #     :param policy: Policy Network under which to take action
    #     :param X: stacked tensor of shape (4,80,120)
    #     :param epsilon: float probability of sampling a random action
    #     :param env: Gymnasium environment
    #     :return: Randomly sampled action or maximum action under policy network
    #     """
    #     if random.uniform(0, 1) < epsilon:
    #         return env.action_space.sample()
    #     else:
    #         with torch.no_grad():
    #             X = X.unsqueeze(0).to(device)
    #             return policy(X).max(1)[1].view(1, 1).item()

    def get_action(self, policy, s, eps=0.1):
        with torch.no_grad():
            self.seq.pop(0)
            self.seq.append(s)
            if np.random.random() >= eps:
                X = torch.tensor(np.vstack(self.seq).astype(np.float32), device=device, dtype=torch.float)
                a = policy(X.unsqueeze(0))
                # a = a[:, -1, :]  # select last element of seq
                a = a.max(1)[1]
                return a.item()
            else:
                return env.action_space.sample()

    # def initialize_measuring_states(self, env):
    #     """
    #     Randomly samples 200 states by taking random actions
    #     :param env: Gymnasium environment
    #     :return: list of states that were visited by random walk
    #     """
    #     measuring_states = []
    #     env.reset()
    #     for i in range(200):
    #         action = env.action_space.sample()
    #         env.step(action)
    #         img = env.render()
    #         img = transforms.ToTensor()(img)
    #         measuring_states.append(transform(img))
    #     env.reset()
    #     return measuring_states

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

            # up to 30 no-op actions
            noop = random.randint(0, 30)
            for i in range(noop):
                action = env.action_space.sample()
                env.step(action)

            rewards = 0

            for i in range(0, self.max_steps):      # max episode length 10000
                img = env.render()
                img = transforms.ToTensor()(img)
                X = transform(img).to(device)

                # action = self.epsilon_greedy_policy(policy, X, self.eval_epsilon, env)
                action = self.get_action(policy, X, self.eval_epsilon)
                _, reward, terminated, _, _ = env.step(action)
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

        # keep track of total steps and rewards
        total_steps = 0
        total_rewards = []
        total_steps_list = []
        evaluations = []

        # initialize states in which Q value is measured every X episodes to track progress
        # measuring_states = self.initialize_measuring_states(env)
        # q_measures = []

        # Initialize Experience Memory
        experience_memory = RecurrentExperienceMemory(self.replay_size)

        # initialize policy (and target) network
        if self.dueling:
            policy = DRQN(env.action_space.n).to(device)
            if self.fixed_target:
                target = DRQN(env.action_space.n).to(device)
                target.load_state_dict(policy.state_dict())
                target.eval()
        else:
            policy = DRQN(env.action_space.n).to(device)
            if self.fixed_target:
                target = DRQN(env.action_space.n).to(device)
                target.load_state_dict(policy.state_dict())
                target.eval()

        # Best values found during evaluation
        best_reward = - float('inf')
        best_policy = policy.state_dict()

        optimizer = torch.optim.Adadelta(policy.parameters(), lr=0.1)

        for episode in range(self.n_training_episodes):
            steps = 0
            rewards = 0

            env.reset()

            # up to 30 no-op actions
            noop = random.randint(0, 30)
            for i in range(noop):
                action = env.action_space.sample()
                env.step(action)

            # get starting state as input tensor
            img = env.render()
            img = transforms.ToTensor()(img)
            X = transform(img)

            while True:
                # linear epsilon decay based on steps
                epsilon = max(self.epsilon_max - ((self.epsilon_max - self.epsilon_min)/self.epsilon_frame) *
                              total_steps, self.epsilon_min)

                # Choose the action At using epsilon greedy policy
                # action = self.epsilon_greedy_policy(policy, X, epsilon, env)
                action = self.get_action(policy, X, epsilon)
                # take action
                _, reward, terminated, _, _ = env.step(action)
                img = env.render()
                img = transforms.ToTensor()(img)
                X_new = transform(img) #if not terminated else None

                experience_memory.add((X, action, reward, X_new, terminated))

                steps += 1
                total_steps += 1

                if len(experience_memory) > self.batch_size * self.sequence_length:
                    # experiences = experience_memory.sample(self.batch_size)
                    # states, actions, _rewards, next_states, terminations = (i for i in zip(*experiences))
                    # a = (torch.tensor(actions).long().unsqueeze(dim=1)).to(device)
                    # r = torch.tensor(_rewards).unsqueeze(dim=1).to(device)
                    # states = np.vstack(states).astype(np.float32)
                    # states = torch.from_numpy(states)
                    # next_states = np.vstack(next_states).astype(np.float32)
                    # next_states = torch.from_numpy(next_states)
                    # states = torch.reshape(states, (self.batch_size, 1, 84, 84)).to(device)  # 80,120
                    # next_states = torch.reshape(next_states, (self.batch_size, 1, 84, 84)).to(device)
                    # mask = [i for i, x in enumerate(terminations) if not x]  # get all non-final states
                    #
                    # reward = torch.tensor([reward]).to(device)
                    # action = torch.tensor([action]).unsqueeze(0).to(device)
                    # state = X.unsqueeze(0)
                    # new_state = X_new.unsqueeze(0)
                    #
                    # steps += 1
                    # total_steps += 1
                    #
                    # state_action_values = policy(states).gather(1, a)
                    #
                    # next_state_values = torch.zeros(self.batch_size, device=device)
                    #
                    # # update network
                    # if self.double:
                    #     max_next_action = policy(next_states).max(1)[1].view(-1, 1)
                    #     next_state_values[mask] = target(next_states[mask]).gather(1, max_next_action[mask]).squeeze(1)
                    # elif self.fixed_target:
                    #     next_state_values[mask] = target(next_states[mask]).max(1)[0].detach()
                    # else:
                    #     next_state_values[mask] = policy(next_states[mask]).max(1)[0].detach()
                    # # Compute the expected Q values
                    # expected_state_action_values = (next_state_values * self.gamma) + r.squeeze(1)
                    shape = (self.batch_size, self.sequence_length, 84, 84)

                    transitions = experience_memory.sample(self.batch_size)
                    # batch_state, batch_action, batch_reward, batch_next_state, done = (i for i in zip(*transitions))
                    # print(len(list(zip(*[zip(*transition) for transition in transitions]))))
                    batch_state, batch_action, batch_reward, batch_next_state, done = zip(*[zip(*t) for t in transitions])
                    # print(torch.tensor(batch_state[0]))
                    batch_state = torch.tensor(np.vstack([np.vstack(batch).astype(np.float32) for batch in batch_state])\
                                               .astype(np.float32), device=device).view(shape)
                    # batch_state = np.vstack(batch_state).astype(np.float32)
                    # batch_state = torch.from_numpy(batch_state)
                    # batch_next_state = np.vstack(batch_next_state).astype(np.float32)
                    batch_next_state = np.vstack([np.vstack(batch).astype(np.float32) for batch in batch_next_state])\
                                                  .astype(np.float32)
                    # batch_next_state = torch.from_numpy(batch_next_state)

                    # batch_state = torch.tensor(batch_state, device=device, dtype=torch.float).view(shape)
                    batch_action = torch.tensor(batch_action, device=device, dtype=torch.long).view(
                        self.batch_size, self.sequence_length)
                    batch_reward = torch.tensor(batch_reward, device=device, dtype=torch.float).view(
                        self.batch_size, self.sequence_length)
                    non_final_mask = torch.tensor([not i for i in done], device=device, dtype=torch.bool)#.view(32,10)
                    # get set of next states for end of each sequence
                    batch_next_state = tuple([batch_next_state[i] for i in range(len(batch_next_state)) if
                                              (i + 1) % self.sequence_length == 0])

                    # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)),
                    #                               device=device, dtype=torch.uint8)
                    # try:  # sometimes all next states are false, especially with nstep returns
                    #     non_final_next_states = torch.tensor(np.array([s for s in batch_next_state if s is not None]),
                    #                                          device=device, dtype=torch.float).unsqueeze(dim=1)
                    #     non_final_next_states = torch.cat([batch_next_state[non_final_mask, 1:, :], non_final_next_states],
                    #                                       dim=1)
                    #     empty_next_state_values = False
                    # except:
                    #     non_final_next_states = None
                    #     empty_next_state_values = True
                    if not any(done):
                        empty_next_state_values = True
                    else:
                        empty_next_state_values = False
                    non_final_next_states = torch.tensor(np.array([s for s in batch_next_state]), device=device)#[non_final_mask]

                    print(non_final_next_states.shape)
                    # non_final_next_states = non_final_next_states[non_final_mask]
                    non_final_next_states = torch.tensor(np.array([v.cpu() for s,v in enumerate(non_final_next_states) if not non_final_mask[s]]), device=device)
                    print(non_final_next_states.shape)

                    # estimate
                    current_q_values = policy(batch_state.squeeze())
                    current_q_values = current_q_values.gather(1, batch_action).squeeze()

                    # target
                    with torch.no_grad():
                        max_next_q_values = torch.zeros((self.batch_size, self.sequence_length), device=device,
                                                            dtype=torch.float)
                        if not empty_next_state_values:
                            max_next, _ = target(non_final_next_states)
                            max_next_q_values[non_final_mask] = max_next.max(dim=2)[0]
                        expected_q_values = batch_reward + (self.gamma * max_next_q_values)

                    # loss = torch.nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1)).squeeze()
                    loss = torch.nn.MSELoss()(current_q_values, expected_q_values)

                    loss = loss.mean()

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
                if terminated or steps >= self.max_steps-1:  # or truncated:
                    # Track rewards
                    total_rewards.append(rewards)
                    total_steps_list.append(steps)

                    # reset sequence
                    self.seq = [np.zeros(1) for _ in range(self.sequence_length)]

                    # # measure Q values in selected states
                    # Q_states = torch.stack(measuring_states).to(device)
                    # Q_states = torch.unique(Q_states, dim=0, sorted=False)  # eliminate duplicate states
                    # with torch.no_grad():
                    #     q_measures.append(torch.mean(policy(Q_states).max(1)[0]).item())

                    # Evaluate current policy and save optimal policy weights
                    if episode == self.n_training_episodes-1 or (episode > 0 and episode % self.eval_every == 0):
                        eval_reward = self.eval(policy, env)
                        if eval_reward > best_reward:
                            best_reward = eval_reward
                            best_policy = policy.state_dict()
                        print(f"Evaluation: {int(episode/self.eval_every)}\t average reward: {eval_reward}")
                        evaluations.append(eval_reward)

                    # print training information
                    if self.debug:
                        print(f"episode: {episode + 1:03d}\t steps: {steps + 1:05d}\t total steps:"
                              f"{total_steps + 1:06d}\t epsilon: {epsilon:.2f}")#\t average Q: {q_measures[-1]:.3f}")
                    break

                if self.fixed_target:
                    # copy policy network weights to target net every copy_target steps
                    if total_steps % self.copy_target <= 4:
                        target.load_state_dict(policy.state_dict())

        return total_rewards, total_steps_list, best_policy, evaluations


# Hyperparameters
n_training_episodes = 1000
gamma = 0.99
learning_rate = 0.1  # 0.1
max_training_steps = 10000

# Exploration parameters
epsilon_max = 1
epsilon_min = 0.1
epsilon_frame = 100000

# replay memory parameters
replay_size = 100000
batch_size = 32

# fixed target network
fixed_target = True
copy_target = 10000

debug = True

transform = scale_and_resize()

env = gym.make('MountainCar-v0', render_mode='rgb_array')

car = TrainRecurrentMountainCar(n_training_episodes=n_training_episodes, gamma=gamma, learning_rate=learning_rate,
                       epsilon_max=epsilon_max, epsilon_min=epsilon_min,
                       max_steps=max_training_steps, batch_size=batch_size, fixed_target=fixed_target,
                       copy_target=copy_target, debug=debug, env=env, epsilon_frame=epsilon_frame)

total_rewards, total_steps_list, q_measures, best_policy, evaluations = car.train()

# save best policy as well as steps and q measures
torch.save(best_policy, 'data/DRQN_final.pth')
np.savetxt(f'data/steps_DRQN_txt', total_steps_list)
# np.savetxt(f'data/q_values_DRQN.txt', q_measures)
np.savetxt(f'data/eval_DRQN.txt', evaluations)

# Plot steps per episode
plt.plot(np.arange(len(total_steps_list)) + 1, total_steps_list, zorder=0, label='training')
x = np.arange(50, n_training_episodes+1, 50)
plt.scatter(x, [-e for e in evaluations], color='r', marker='x', zorder=1, label='evaluations')
N = 10
steps_mean = running_mean(total_steps_list, N)
plt.plot(np.arange(len(steps_mean)) + 1, steps_mean, zorder=0, label='running average')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode - DRQN')
plt.savefig('plots/steps_DRQN.png')
plt.close()

# Plot q measures per episode
# plt.plot(np.arange(len(q_measures)) + 1, q_measures)
# plt.xlabel('Episode')
# plt.ylabel('Average Q')
# plt.title('Average Q measure over sampled states')
# plt.savefig('plots/q_measures_DRQN.png')
# plt.close()
