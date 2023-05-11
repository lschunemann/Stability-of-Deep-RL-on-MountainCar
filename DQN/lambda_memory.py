import numpy as np


def make_replay_memory(capacity, history_len, discount, cache_size, block_size, priority, lambd):
    lambd = lambd
    return LambdaReplayMemory(capacity, history_len, discount, cache_size, block_size, priority, lambd, use_watkins=True)


class ReplayMemory:
    def __init__(self, capacity, history_len, discount, cache_size, block_size, priority):
        assert (cache_size % block_size) == 0
        # Extra samples to fit exactly `capacity` (overlapping) blocks
        self.capacity = capacity + (history_len - 1) + block_size
        self.history_len = history_len
        self.discount = discount
        self.num_samples = 0

        self.cache_size = cache_size
        self.block_size = block_size
        self.priority = priority
        self.refresh_func = None

        # Main variables for memory
        self.obs = None  # Allocated dynamically once shape/dtype are known
        self.actions = np.empty([self.capacity], dtype=np.int32)
        self.rewards = np.empty([self.capacity], dtype=np.float32)
        self.dones = np.empty([self.capacity], dtype=np.bool)
        self.next = 0  # Points to next transition to be overwritten

        # Auxiliary buffers for the cache -- pre-allocated to smooth memory usage
        self.cached_states = None  # Allocated dynamically once shape/dtype are known
        self.cached_actions = np.empty([self.cache_size], dtype=np.int32)
        self.cached_returns = np.empty([self.cache_size], dtype=np.float32)
        self.cached_errors = np.empty([self.cache_size], dtype=np.float32)
        self.cached_indices = np.empty([self.cache_size], dtype=np.int32)

    def register_refresh_func(self, f):
        assert self.refresh_func is None
        self.refresh_func = f

    def sample(self, batch_size):
        start = self.batch_counter * batch_size
        end = start + batch_size
        indices = self.cached_indices[start:end]

        state_batch = self.cached_states[indices]
        action_batch = self.cached_actions[indices]
        return_batch = self.cached_returns[indices]

        self.batch_counter += 1

        return np.array(state_batch), np.array(action_batch), np.array(return_batch)

    def encode_recent_observation(self):
        i = self.len()
        return self._encode_observation(i)

    def _encode_observation(self, i):
        i = self._align(i)

        # Start with blank observations except the last
        state = np.zeros([self.history_len, *self.obs[0].shape], dtype=self.obs[0].dtype)
        state[-1] = self.obs[i]

        # Fill-in backwards, break if we reach a terminal state
        for j in range(1, min(self.history_len, self.len())):
            if self.dones[i - j]:
                break
            state[-1 - j] = self.obs[i - j]

        return state

    def _align(self, i):
        # Make relative to pointer when full
        if not self.full(): return i
        return (i + self.next) % self.capacity

    def store_obs(self, obs):
        if self.obs is None:
            self.obs = np.empty([self.capacity, *obs.shape], dtype=obs.dtype)
        if self.cached_states is None:
            self.cached_states = np.empty([self.cache_size, self.history_len, *obs.shape], dtype=obs.dtype)
        self.obs[self.next] = obs

    def store_effect(self, action, reward, done):
        self.actions[self.next] = action
        self.rewards[self.next] = reward
        self.dones[self.next] = done

        self.next = (self.next + 1) % self.capacity
        self.num_samples = min(self.capacity, self.num_samples + 1)

    def len(self):
        return self.num_samples

    def full(self):
        return self.len() == self.capacity

    def refresh(self, train_frac):
        # Reset batch counter
        self.batch_counter = 0

        # Sample blocks until we have enough data
        num_blocks = self.cache_size // self.block_size
        block_ids = self._sample_block_ids(num_blocks)

        self._refresh(train_frac, block_ids)  # Separate function for unit testing

    def _refresh(self, train_frac, block_ids):
        # Refresh the blocks we sampled and load them into the cache
        for k, i in enumerate(block_ids):
            states = self._extract_block(None, i, states=True)
            actions = self._extract_block(self.actions, i)
            rewards = self._extract_block(self.rewards, i)
            dones = self._extract_block(self.dones, i)

            max_qvalues, mask, onpolicy_qvalues = self.refresh_func(states, actions)
            returns = self._calculate_returns(rewards, max_qvalues, dones, mask)
            errors = np.abs(returns - onpolicy_qvalues)

            start = self.block_size * k
            end = start + self.block_size

            self.cached_states[start:end] = states[:-1]
            self.cached_actions[start:end] = actions
            self.cached_returns[start:end] = returns
            self.cached_errors[start:end] = errors

        # Prioritize samples
        distr = self._prioritized_distribution(self.cached_errors, train_frac)
        self.cached_indices = np.random.choice(self.cache_size, size=self.cache_size, replace=True, p=distr)

    def _sample_block_ids(self, n):
        return np.random.randint(self.history_len - 1, self.len() - self.block_size, size=n)

    def _extract_block(self, a, start, states=False):
        end = start + self.block_size
        if states:
            assert a is None
            return np.array([self._encode_observation(i) for i in range(start, end + 1)])
        return a[self._align(np.arange(start, end))]

    def _prioritized_distribution(self, errors, train_frac):
        # Start with the uniform distribution.
        distr = np.ones_like(errors) / self.cache_size
        # Adjust the probabilities based on whether their corresponding errors lie above/below the median.
        p = self.priority_now(train_frac)
        med = np.median(errors)
        distr[errors > med] *= (1.0 + p)
        distr[errors < med] *= (1.0 - p)
        # Note that if the error was identically equal to the median, its probability was not adjusted;
        # this is the correct behavior to guarantee the probabilities sum to 1.
        # However, due to floating point errors, we still need to re-normalize the distribution here:
        return distr / distr.sum()

    def priority_now(self, train_frac):
        return self.priority * (1.0 - train_frac)

    def _calculate_returns(self, rewards, qvalues, dones, mask):
        raise NotImplementedError


class LambdaReplayMemory(ReplayMemory):
    def __init__(self, capacity, history_len, discount, cache_size, block_size, priority, lambd, use_watkins):
        self.lambd = lambd
        self.use_watkins = use_watkins
        super().__init__(capacity, history_len, discount, cache_size, block_size, priority)

    def _calculate_returns(self, rewards, qvalues, dones, mask):
        if not self.use_watkins:
            mask = np.ones_like(qvalues)
        return calculate_lambda_returns(rewards, qvalues, dones, mask, self.discount, self.lambd)


def calculate_lambda_returns(rewards, qvalues, dones, mask, discount, lambd):
    dones = dones.astype(np.float32)
    qvalues[-1] *= (1.0 - dones[-1])
    lambda_returns = rewards + (discount * qvalues[1:])
    for i in reversed(range(len(rewards) - 1)):
        a = lambda_returns[i] + (discount * lambd * mask[i]) * (lambda_returns[i + 1] - qvalues[i + 1])
        b = rewards[i]
        lambda_returns[i] = (1.0 - dones[i]) * a + dones[i] * b
    return lambda_returns
