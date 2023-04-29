import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from models import DQN_square, DQN_dueling
import gymnasium as gym
import random
from helper_DQN import scale_and_resize, ReplayMemory
from torchrl.data import ListStorage, PrioritizedReplayBuffer

device = torch.device("cuda")
transform = scale_and_resize()


class TrainMountainCar:
    def __init__(self, n_training_episodes=200, gamma=0.99, learning_rate=0.1, epsilon_max=0.5,
                 epsilon_min=0.05, decay_rate=0.005, max_steps=10000, batch_size=32, fixed_target=False,
                 copy_target=10000, replay_size=100000, double=False, dueling=False, prioritized=False, debug=False,
                 eval_epsilon=0.05, eval_episodes=25, eval_every=50):
        self.n_training_episodes = n_training_episodes
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.fixed_target = fixed_target
        self.copy_target = copy_target
        self.replay_size = replay_size
        self.double = double
        self.dueling = dueling
        self.debug = debug
        self.prioritized = prioritized
        self.eval_epsilon = eval_epsilon
        self.eval_episodes = eval_episodes
        self.eval_every = eval_every

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
            terminated = stop if stop else terminated       # if one of the visited stated is terminal, return terminal
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

        # keep track of total steps and rewards (start total steps as -1 because of initial frame stack)
        total_steps = 0
        total_rewards = []
        total_steps_list = []

        # initialize states in which Q value is measured every X episodes to track progress
        measuring_states = self.initialize_measuring_states(env)
        q_measures = []

        if self.prioritized:
            experience_memory = PrioritizedReplayBuffer(alpha=0.7, beta=0.5, storage=ListStorage(self.replay_size))
        else:
            experience_memory = ReplayMemory(self.replay_size)

        # initialize policy (and target) network
        if self.dueling:
            policy = DQN_dueling(env.action_space.n).to(device)
            if self.fixed_target:
                target = DQN_dueling(env.action_space.n).to(device)
                target.load_state_dict(policy.state_dict())
                target.eval()
        else:
            policy = DQN_square(env.action_space.n).to(device)
            if self.fixed_target:
                target = DQN_square(env.action_space.n).to(device)
                target.load_state_dict(policy.state_dict())
                target.eval()

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
            X = torch.stack(stacked_images)

            while True:
                # epsilon decay based on steps
                # epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.decay_rate * total_steps)
                epsilon = max(self.epsilon_max - ((self.epsilon_max - self.epsilon_min)/500000) * total_steps, self.epsilon_min)      # linear decay
                # epsilon decay based on episodes
                # epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-decay_rate * 100 * total_steps)

                # Choose the action At using epsilon greedy policy
                action = self.epsilon_greedy_policy(policy, X, epsilon, env)
                # take action
                stacked_images, reward, terminated = self.prepare_images(env, action)
                # _, reward, terminated, truncated, _ = env.step(action)
                reward = torch.tensor([reward])

                # update image stack with new state
                # stacked_images, X_new = self.update_img_stack(env, stacked_images)
                X_new = torch.stack(stacked_images)

                # update experience memory
                experience_memory.add([X, action, reward, X_new], terminated)

                # update current input to be next input
                X = X_new

                steps += 4
                total_steps += 4

                if len(experience_memory) > self.batch_size:
                    # extract old states, actions, rewards and new states from ReplayMemory
                    if self.prioritized:
                        experiences, info = experience_memory.sample(self.batch_size)
                    else:
                        experiences = experience_memory.sample(self.batch_size)
                    experiences = np.asarray(experiences, dtype=object)
                    experience = experiences[:, 0]
                    experience = np.asarray([np.array(i, dtype=object) for i in experience])
                    states = experience[:, 0]
                    a = experience[:, 1]
                    r = experience[:, 2]
                    next_states = experience[:, 3]
                    terminations = experiences[:, 1]
                    mask = [i for i, x in enumerate(terminations) if not x]     # get all non-final states

                    # change states and rewards back to tensors
                    states = np.vstack(states).astype(np.float32)
                    states = torch.from_numpy(states)
                    next_states = np.vstack(next_states).astype(np.float32)
                    next_states = torch.from_numpy(next_states)
                    r = np.vstack(r).astype(np.float32)
                    r = torch.from_numpy(r).to(device)
                    a = np.vstack(a).astype(np.float32)
                    a = torch.from_numpy(a).to(device)
                    states = torch.reshape(states, (32, 4, 84, 84)).to(device)      # 80,120
                    next_states = torch.reshape(next_states, (32, 4, 84, 84)).to(device)

                    # calculate target
                    if self.double:     # DDQN
                        act = target(states).max(1)[1].unsqueeze(1)
                    else:
                        act = a
                    state_action_values = policy(states).gather(1, act.type(torch.int64))
                    # print(torch.mean(policy(states).max(1)[0]).item())
                    # target = r + self.gamma * np.argmax(values(x_new))
                    # update network
                    next_state_values = torch.zeros(self.batch_size, device=device)
                    if self.fixed_target:
                        next_state_values[mask] = target(next_states[mask]).max(1)[0].detach()
                    else:
                        next_state_values[mask] = policy(next_states[mask]).max(1)[0].detach()
                    # print(torch.mean(target(next_states[mask]).max(1)[0]).item())
                    # Compute the expected Q values
                    expected_state_action_values = (next_state_values * self.gamma) + r.squeeze(1)
                    # print(torch.mean(expected_state_action_values).item())

                    if self.prioritized:
                        diff = expected_state_action_values.unsqueeze(1) - state_action_values
                        experience_memory.update_priority(info['index'], diff.detach().squeeze().abs().numpy().tolist())

                        loss = torch.nn.MSELoss()(state_action_values,
                                                  expected_state_action_values.unsqueeze(1)).squeeze() * info['_weight']
                    else:
                        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
                        # loss = torch.nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1)).squeeze()
                    # Optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    # for param in policy.parameters():
                    #     param.grad.data.clamp_(-1, 1)
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.)
                    optimizer.step()

                # Update total reward
                rewards += reward

                # If done, finish the episode
                if terminated:  # or truncated:
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

                    # print training information
                    if self.debug:
                        print(f"episode: {episode + 1:03d}\t steps: {steps + 1:05d}\t total steps:"
                              f"{total_steps + 1:06d}\t epsilon: {epsilon:.2f}\t average Q: {q_measures[-1]:.3f}")
                    break

                if self.fixed_target:
                    # copy policy network weights to target net every copy_target steps
                    if total_steps % self.copy_target <= 4:
                        target.load_state_dict(policy.state_dict())
                        # save policy net weights for evaluation
                        # if total_steps % (self.copy_target * 10) <= 4:
                        #     if self.prioritized:
                        #         torch.save(policy.state_dict(), f'data/DDQN_prioritized_{total_steps}.pth')
                        #     elif self.double:
                        #         torch.save(policy.state_dict(), f'data/DDQN_{total_steps}.pth')
                        #     else:
                        #         torch.save(policy.state_dict(), f'data/DQN_paper_fixed_{total_steps}.pth')

        return total_rewards, total_steps_list, q_measures, best_policy

