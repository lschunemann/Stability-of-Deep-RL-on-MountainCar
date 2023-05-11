import random

import gymnasium as gym
import torch
from helper_DQN import transform_to_grayscale_and_normalize, scale_and_resize
import torchvision.transforms as transforms
from models import DQN_square, NoisyNet, NoisyNet_Dueling
import imageio

model_name = "DDQN_prioritized"

# load model
model = DQN_square(3)
model.load_state_dict(torch.load(f"data/{model_name}.pth"))
model.eval()

# initialize environment
env = gym.make("MountainCar-v0", render_mode='rgb_array')

env.reset()
stacked_images = []
img = env.render()
stacked_images.append(transform_to_grayscale_and_normalize(img))

steps = 4

# save video
frames = []

transform = scale_and_resize()

action = env.action_space.sample()

stacked_images = []
for i in range(4):
    _, reward, terminated, _, _ = env.step(action)
    img = env.render()
    frames.append(img)
    img = transforms.ToTensor()(img)
    stacked_images.append(torch.squeeze(transform(img)))

while True:
    X = torch.stack(stacked_images).unsqueeze(0)

    if random.uniform(0,1) < 0.05:
        action = env.action_space.sample()
    else:
        action = model(X).max(1)[1].view(1, 1).item()

    # take action
    _, _, terminated, truncated, _ = env.step(action)

    # update image stack with new state
    stacked_images = []
    for i in range(4):
        _, reward, terminated, _, _ = env.step(action)
        img = env.render()
        frames.append(img)
        img = transforms.ToTensor()(img)
        stacked_images.append(torch.squeeze(transform(img)))

    steps += 4

    if terminated or steps > 2000:  #truncated:
        print(f"finished in {steps} steps")
        imageio.mimsave(f'video/replay_{model_name}.gif', frames, fps=40)
        break
