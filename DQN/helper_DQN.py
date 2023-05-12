import collections

import torchvision
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import random


def transform_to_grayscale_and_normalize(img: np.ndarray):
    """
    transforms an np array to grayscale, normalizes it and converts it to a Pytorch tensor
    :param img: np array
    :return: Torch tensor
    """
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img)
    img = torchvision.transforms.functional.rgb_to_grayscale(img, 1) / 255
    img = torch.squeeze(img)
    return img


class SquarePad(nn.Module):
    def __call__(self, image):
        w, h = 600, 400
        vp = int((w - h) / 2)
        padding = (0, vp, 0, vp)
        return F.pad(image, padding, 0, 'constant')


def scale_and_resize():
    """
    resizes the image to 80x120 and transforms it to grayscale
    :return: Torch tensor
    """
    transform = torch.nn.Sequential(
        SquarePad(),
        # transforms.Resize((80, 120)),
        transforms.Resize((84, 84)),
        transforms.Grayscale(1),
        # transforms.Normalize([0.5], [0.5])
    )
    return transform


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


class ReplayMemory:
    def __init__(self, replay_size=100000):
        self.replay = []
        self.replay_size = replay_size

    def add(self, experience: list, terminated: bool):
        """
        pushes an experience to the Replay Memory by updating the interior representation.
        :param experience: list of old state, reward, action and new state to be added to memory
        :param terminated: whether the new state is a final state
        """
        if len(self.replay) > self.replay_size:
            self.replay = self.replay[1:]
        self.replay.append([experience, terminated])

    def sample(self, batch_size: int):
        """
        sample batch_size number of experiences from the Replay Memory
        :param batch_size: size of sample
        :return:
        """
        return random.sample(self.replay, batch_size)

    def __len__(self):
        return len(self.replay)


Experience = collections.namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class PrioritizedExperienceReplayBuffer:

    def __init__(self, batch_size: int, buffer_size: int, alpha: float = 0.0,
                 random_state: np.random.RandomState = None) -> None:

        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._buffer_length = 0  # current number of prioritized experience tuples in buffer
        self._buffer = np.empty(self._buffer_size, dtype=[("priority", np.float32), ("experience", Experience)])
        self._alpha = alpha
        self._random_state = np.random.RandomState() if random_state is None else random_state

    def __len__(self) -> int:
        return self._buffer_length

    def add(self, experience: Experience) -> None:
        priority = 1.0 if self.is_empty() else self._buffer["priority"].max()
        if self.is_full():
            if priority > self._buffer["priority"].min():
                idx = self._buffer["priority"].argmin()
                self._buffer[idx] = (priority, experience)
            else:
                pass  # low priority experiences should not be included in buffer
        else:
            self._buffer[self._buffer_length] = (priority, experience)
            self._buffer_length += 1

    def is_empty(self) -> bool:
        return self._buffer_length == 0

    def is_full(self) -> bool:
        return self._buffer_length == self._buffer_size

    def sample(self, beta: float):
        # use sampling scheme to determine which experiences to use for learning
        ps = self._buffer[:self._buffer_length]["priority"]
        sampling_probs = ps ** self._alpha / np.sum(ps ** self._alpha)
        idxs = self._random_state.choice(np.arange(ps.size),
                                         size=self._batch_size,
                                         replace=True,
                                         p=sampling_probs)

        # select the experiences and compute sampling weights
        experiences = self._buffer["experience"][idxs]
        weights = (self._buffer_length * sampling_probs[idxs]) ** -beta
        normalized_weights = weights / weights.max()

        return idxs, experiences, normalized_weights

    def update_priority(self, idxs: np.array, priorities: np.array) -> None:
        self._buffer["priority"][idxs] = priorities


class ExperienceMemory:
    def __init__(self, replay_size=100000):
        self.replay_size = replay_size
        self.replay = []

    def add(self, experience: Experience):
        """
        pushes an experience to the Replay Memory by updating the interior representation.
        :param experience: list of old state, reward, action and new state to be added to memory
        """
        if len(self.replay) > self.replay_size:
            self.replay = self.replay[1:]
        self.replay.append(experience)

    def sample(self, batch_size: int):
        """
        sample batch_size number of experiences from the Replay Memory
        :param batch_size: size of sample
        :return:
        """
        return random.sample(self.replay, batch_size)

    def __len__(self):
        return len(self.replay)


class RecurrentExperienceMemory:
    def __init__(self, capacity, sequence_length=10):
        self.capacity = capacity
        self.memory = []
        self.seq_length = sequence_length

    def add(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    # def sample(self, batch_size):
    #     finish = random.sample(range(0, len(self.memory)), batch_size)
    #     begin = [x - self.seq_length for x in finish]
    #     samp = []
    #     for start, end in zip(begin, finish):
    #         # correct for sampling near beginning
    #         final = self.memory[max(start + 1, 0):end + 1]
    #
    #         # correct for sampling across episodes
    #         for i in range(len(final) - 2, -1, -1):
    #             if final[i][3] is None:
    #                 final = final[i + 1:]
    #                 break
    #
    #         # pad beginning to account for corrections
    #         while (len(final) < self.seq_length):
    #             final = [(np.zeros_like(self.memory[0][0].cpu()), 0, 0, np.zeros_like(self.memory[0][3].cpu()))] + final
    #
    #         samp += final
    #
    #     # returns flattened version
    #     return samp, None, None

    def sample(self, batch_size):
        idxs = random.sample(range(10, len(self.memory)), batch_size)
        sampl = []
        for idx in idxs:
            sampl.append(self.memory[idx-self.seq_length:idx])
        return sampl

    def __len__(self):
        return len(self.memory)
