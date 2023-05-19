import collections

import torch
import torchvision.transforms as transforms
import numpy as np
from models import DQN_square, DQN_dueling, NoisyNet_Dueling, NoisyNet, Categorical_DQN, Quantile_DQN, Rainbow_DQN
import gymnasium as gym
import random
from helper_DQN import scale_and_resize, ExperienceMemory, PrioritizedExperienceReplayBuffer

device = torch.device("cuda")
transform = scale_and_resize()

Experience = collections.namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class TrainMountainCar:
    def __init__(self, n_training_episodes=1000, gamma=0.99, learning_rate=0.1, epsilon_max=1,
                 epsilon_min=0.05, max_steps=10000, batch_size=32, fixed_target=False,
                 copy_target=10000, replay_size=100000, double=False, dueling=False, prioritized=False, debug=False,
                 eval_epsilon=None, eval_episodes=10, eval_every=50, noisy=False, distributional=False,
                 epsilon_frame=1000000, quantiles=51, quantile=False, rainbow=False, min_memory=50000):
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
        self.prioritized = prioritized
        self.eval_epsilon = epsilon_min if eval_epsilon is None else eval_epsilon
        self.eval_episodes = eval_episodes
        self.eval_every = eval_every
        self.noisy = noisy
        self.distributional = distributional
        self.atoms = 51
        self.v_max = 10
        self.v_min = -10
        self.supports = torch.linspace(self.v_min, self.v_max, self.atoms).view(1, 1, self.atoms).to(device)
        self.delta = (self.v_max - self.v_min) / (self.atoms - 1)
        self.num_quantiles = quantiles
        self.cumulative_density = torch.tensor((2 * np.arange(self.num_quantiles) + 1) / (2.0 * self.num_quantiles),
                                               device=device, dtype=torch.float)
        self.quantile_weight = 1.0 / self.num_quantiles
        self.quantile = quantile
        self.rainbow = rainbow
        self.min_memory = min_memory

    def epsilon_greedy_policy(self, policy: torch.nn.Module, X, epsilon: float, env: gym.envs):
        """
        Samples a random action with probability epsilon and picks the maximum action under policy network otherwise.
        :param policy: Policy Network under which to take action
        :param X: stacked tensor of shape (4,84,84)
        :param epsilon: float probability of sampling a random action
        :param env: Gymnasium environment
        :return: Randomly sampled action or maximum action under policy network
        """
        if self.rainbow:
            with torch.no_grad():
                X = X.unsqueeze(0).to(device)
                policy.sample_noise()
                a = policy(X) * self.supports
                a = a.sum(dim=2).max(1)[1].view(1, 1)
                return a.item()
        elif self.noisy:
            with torch.no_grad():
                X = X.unsqueeze(0).to(device)
                policy.sample_noise()
                return policy(X).max(1)[1].view(1, 1).item()
        elif random.uniform(0, 1) < epsilon:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                X = X.unsqueeze(0).to(device)
                if self.quantile:
                    a = (policy(X) * self.quantile_weight).sum(dim=2).max(dim=1)[1]
                    return a.item()
                elif self.distributional:
                    a = policy(X) * self.supports
                    a = a.sum(dim=2).max(1)[1].view(1, 1)
                    return a.item()
                else:
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
            if stop: terminated = True       # if one of the visited states is terminal, return terminal
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

    def init_networks(self, env):
        if self.rainbow:
            policy = Rainbow_DQN(env.action_space.n, self.atoms).to(device)
            target = Rainbow_DQN(env.action_space.n, self.atoms).to(device)
            target.load_state_dict(policy.state_dict())
            target.eval()
        elif self.quantile:
            policy = Quantile_DQN(env.action_space.n, self.num_quantiles).to(device)
            target = Quantile_DQN(env.action_space.n, self.num_quantiles).to(device)
            target.load_state_dict(policy.state_dict())
            target.eval()
        elif self.distributional:
            policy = Categorical_DQN(env.action_space.n, self.atoms).to(device)
            target = Categorical_DQN(env.action_space.n, self.atoms).to(device)
            target.load_state_dict(policy.state_dict())
            target.eval()
        elif self.noisy:
            if self.dueling:
                policy = NoisyNet_Dueling(env.action_space.n).to(device)
                if self.fixed_target:
                    target = NoisyNet_Dueling(env.action_space.n).to(device)
                    target.load_state_dict(policy.state_dict())
                    target.eval()
            else:
                policy = NoisyNet(env.action_space.n).to(device)
                if self.fixed_target:
                    target = NoisyNet(env.action_space.n).to(device)
                    target.load_state_dict(policy.state_dict())
        elif self.dueling:
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
        if 'target' not in locals():
            target = None
        return policy, target

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)

    def compute_loss(self, states, a, r, next_states, terminations, policy, target, mask, experience_memory, idxs=None,
                     weights=None):
        if self.rainbow:
            batch_action = a.unsqueeze(dim=-1).expand(-1, -1, self.atoms)
            batch_reward = r.view(-1, 1, 1)

            empty_next_state_values = not any(terminations)
            non_final_mask = torch.tensor([not i for i in terminations], device=device, dtype=torch.bool)

            # estimate
            policy.sample_noise()
            current_dist = policy(states).gather(1, batch_action).squeeze()

            target_prob = self.projection_distribution(target, states, batch_action, batch_reward,
                                                       next_states[non_final_mask], non_final_mask,
                                                       empty_next_state_values)

            loss = -(target_prob * current_dist.log()).sum(-1)
            experience_memory.update_priority(idxs, loss.detach().squeeze().abs().cpu().numpy().tolist())
            loss = loss * weights

        elif self.quantile:
            batch_action = a.unsqueeze(dim=-1).expand(-1, -1, self.num_quantiles)

            quantiles = policy(states)
            quantiles = quantiles.gather(1, batch_action).squeeze(1)

            empty_next_state_values = not any(terminations)
            non_final_mask = torch.tensor([not i for i in terminations], device=device, dtype=torch.bool)

            quantiles_next = self.next_distribution(r, next_states, next_states[non_final_mask],
                                                    empty_next_state_values, non_final_mask, target)

            diff = quantiles_next.t().unsqueeze(-1) - quantiles.unsqueeze(0)

            loss = self.huber(diff) * torch.abs(self.cumulative_density.view(1, -1) - (diff < 0).to(torch.float))
            loss = loss.transpose(0, 1)
            loss = loss.mean(1).sum(-1)

        elif self.distributional:
            batch_action = a.unsqueeze(dim=-1).expand(-1, -1, self.atoms)
            batch_reward = r.view(-1, 1, 1)
            empty_next_state_values = not any(terminations)

            non_final_mask = torch.tensor([not i for i in terminations], device=device, dtype=torch.bool)

            # estimate
            current_dist = policy(states).gather(1, batch_action).squeeze()

            target_prob = self.projection_distribution(target, states, batch_action, batch_reward,
                                                       next_states[non_final_mask], non_final_mask,
                                                       empty_next_state_values)

            loss = -(target_prob * current_dist.log()).sum(-1)
        else:
            if self.noisy:
                policy.sample_noise()

            state_action_values = policy(states).gather(1, a.type(torch.int64))
            # target = r + self.gamma * np.argmax(values(x_new))
            # update network
            next_state_values = torch.zeros(self.batch_size, device=device)

            if self.noisy:
                target.sample_noise()

            if self.double:
                max_next_action = policy(next_states).max(1)[1].view(-1, 1)
                next_state_values[mask] = target(next_states[mask]).gather(1, max_next_action[mask]).squeeze(1)
            elif self.fixed_target:
                next_state_values[mask] = target(next_states[mask]).max(1)[0].detach()
            else:
                next_state_values[mask] = policy(next_states[mask]).max(1)[0].detach()
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.gamma) + r.squeeze(1)

            if self.prioritized:
                diff = expected_state_action_values.unsqueeze(1) - state_action_values
                experience_memory.update_priority(idxs, diff.cpu().detach().squeeze().abs().numpy().tolist())

                loss = torch.nn.MSELoss()(state_action_values,
                                          expected_state_action_values.unsqueeze(1)).squeeze() * weights
            else:
                # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
                loss = torch.nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1)).squeeze()
        return loss

    def next_distribution(self, r, next_states, non_final_next_states, empty_next_state_values, non_final_mask, target):
        with torch.no_grad():
            quantiles_next = torch.zeros((self.batch_size, self.num_quantiles), device=device, dtype=torch.float)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(target, non_final_next_states)
                quantiles_next[non_final_mask] = target(non_final_next_states).gather(1, max_next_action).squeeze(dim=1)

            quantiles_next = r + (self.gamma * quantiles_next)

        return quantiles_next

    def get_max_next_state_action(self, target, next_states):
        if self.distributional or self.rainbow:
            next_dist = target(next_states) * self.supports
            return next_dist.sum(dim=2).max(1)[1].view(next_states.size(0), 1, 1).expand(-1, -1, self.atoms)
        elif self.quantile:
            next_dist = target(next_states) * self.quantile_weight
            return next_dist.sum(dim=2).max(1)[1].view(next_states.size(0), 1, 1).expand(-1, -1, self.num_quantiles)

    def projection_distribution(self, target, batch_state, batch_action, batch_reward, non_final_next_states,
                                non_final_mask, empty_next_state_values):
        # batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values,\
        #     indices, weights = batch_vars

        with torch.no_grad():
            max_next_dist = torch.zeros((self.batch_size, self.atoms), device=device,
                                        dtype=torch.float) + 1. / self.atoms
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(target, non_final_next_states)
                max_next_dist[non_final_mask] = target(non_final_next_states).gather(1, max_next_action).squeeze()
                max_next_dist = max_next_dist.squeeze()

            Tz = batch_reward.view(-1, 1) + self.gamma * self.supports.view(1, -1) * non_final_mask.to(
                torch.float).view(-1, 1)
            Tz = Tz.clamp(self.v_min, self.v_max)
            b = (Tz - self.v_min) / self.delta
            l = b.floor().to(torch.int64)
            u = b.ceil().to(torch.int64)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            offset = torch.linspace(0, (self.batch_size - 1) * self.atoms, self.batch_size).unsqueeze(dim=1).expand(
                self.batch_size, self.atoms).to(batch_action)
            m = batch_state.new_zeros(self.batch_size, self.atoms)
            m.view(-1).index_add_(0, (l + offset).view(-1),
                                  (max_next_dist * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1),
                                  (max_next_dist * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        return m

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

            for i in range(0, self.max_steps, 4):      # max episode length 10000
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
        td_errors = []

        # beta for prioritized experience replay
        if self.prioritized or self.rainbow:
            beta_start = 0.5
            beta_frames = 1000
            beta_by_frame = lambda total_steps: min(1.0, beta_start + total_steps * (1.0 - beta_start) / beta_frames)

        # initialize states in which Q value is measured every X episodes to track progress
        measuring_states = self.initialize_measuring_states(env)
        q_measures = []

        # initialize Experience Replay memory
        if self.prioritized or self.rainbow:
            # experience_memory = PrioritizedReplayBuffer(alpha=0.7, beta=beta_start, storage=ListStorage(self.replay_size))
            experience_memory = PrioritizedExperienceReplayBuffer(alpha=0.7, batch_size=self.batch_size,
                                                                  buffer_size=self.replay_size)
        else:
            experience_memory = ExperienceMemory(self.replay_size)

        # initialize policy (and target) network
        policy, target = self.init_networks(env)

        # Best values found during evaluation
        best_reward = - float('inf')
        best_policy = policy.state_dict()

        if self.rainbow:
            optimizer = torch.optim.Adam(policy.parameters(), lr=self.learning_rate, weight_decay=0.99, eps=0.00015)
        elif self.distributional or self.quantile:
            optimizer = torch.optim.Adam(policy.parameters(), lr=self.learning_rate, weight_decay=0.99, eps=0.01/self.batch_size)
        else:
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
                # linear epsilon decay based on steps
                epsilon = max(self.epsilon_max - ((self.epsilon_max - self.epsilon_min)/self.epsilon_frame) *
                              total_steps, self.epsilon_min)

                # Choose the action At using epsilon greedy policy
                action = self.epsilon_greedy_policy(policy, X, epsilon, env)
                # take action
                stacked_images, reward, terminated = self.prepare_images(env, action)
                reward = torch.tensor([reward])

                # update image stack with new state
                X_new = torch.stack(stacked_images)

                # update experience memory
                experience_memory.add(Experience(X, action, reward, X_new, terminated))

                # update current state to be next state
                X = X_new

                steps += 4
                total_steps += 4

                # > self.batch_size
                if len(experience_memory) >= self.min_memory:  # start learning after storing some experiences to negate correlation
                    # extract old states, actions, rewards and new states from ReplayMemory
                    if self.prioritized or self.rainbow:
                        beta = beta_by_frame(total_steps)
                        idxs, experiences, weights = experience_memory.sample(beta)
                        states, actions, _rewards, next_states, terminations = (i for i in
                                                                                zip(*experiences))  # (torch.Tensor(vs).to(device) for vs in
                        # zip(*experiences))
                        weights = torch.tensor(weights).to(device)
                    else:
                        experiences = experience_memory.sample(self.batch_size)
                        states, actions, _rewards, next_states, terminations = (i for i in zip(*experiences))

                    # prepare experiences
                    a = (torch.tensor(actions).long().unsqueeze(dim=1)).to(device)
                    r = torch.tensor(_rewards).unsqueeze(dim=1).to(device)
                    states = np.vstack(states).astype(np.float32)
                    states = torch.from_numpy(states)
                    next_states = np.vstack(next_states).astype(np.float32)
                    next_states = torch.from_numpy(next_states)
                    states = torch.reshape(states, (32, 4, 84, 84)).to(device)  # 80,120
                    next_states = torch.reshape(next_states, (32, 4, 84, 84)).to(device)
                    mask = [i for i, x in enumerate(terminations) if not x]  # get all non-final states

                    # calculate loss
                    if 'idxs' not in locals():
                        idxs = None
                    if 'weights' not in locals():
                        weights = None
                    loss = self.compute_loss(states, a, r, next_states, terminations, policy, target, mask,
                                             experience_memory, idxs, weights)

                    loss = loss.mean()
                    td_errors.append(loss.cpu())  # save td errors

                    # Optimize the model
                    optimizer.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.)    # clip gradients
                    optimizer.step()

                    # reset noise
                    if self.noisy or self.rainbow:
                        policy.reset_noise()
                        target.reset_noise()

                # Update total reward
                rewards += reward

                # If done, finish the episode
                if terminated or steps >= self.max_steps-1:  # or truncated:
                    # Track rewards
                    total_rewards.append(rewards)
                    total_steps_list.append(steps)

                    # measure Q values in selected states
                    with torch.no_grad():
                        Z = [torch.stack(measuring_state) for measuring_state in measuring_states]
                        Q_states = torch.stack(Z).to(device)
                        Q_states = torch.unique(Q_states, dim=0, sorted=False)  # eliminate duplicate states
                        q_measures.append(torch.mean(policy(Q_states).max(1)[0]).item())

                    # Evaluate current policy and save optimal policy weights
                    if episode == self.n_training_episodes-1 or (episode > 0 and episode % self.eval_every == 0):
                        eval_reward = self.eval(policy, env)
                        if eval_reward >= best_reward:
                            best_reward = eval_reward
                            best_policy = policy.state_dict()
                        print(f"Evaluation: {int(episode/self.eval_every)}\t average reward: {eval_reward}")
                        evaluations.append(eval_reward)

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

        return total_rewards, total_steps_list, q_measures, best_policy, evaluations, td_errors, policy.state_dict()

