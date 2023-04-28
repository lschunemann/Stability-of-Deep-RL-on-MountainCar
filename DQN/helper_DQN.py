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
