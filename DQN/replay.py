import gymnasium as gym
import torch
from helper_DQN import transform_to_grayscale_and_normalize
from models import DQN
import imageio


# load model
model = DQN(3)
model.load_state_dict(torch.load("data/DQN.pth"))
model.eval()

# initialize environment
env = gym.make("MountainCar-v0", render_mode='rgb_array')

env.reset()
stacked_images = []
img = env.render()
stacked_images.append(transform_to_grayscale_and_normalize(img))

# save video
frames = []

for i in range(3):
    _, _, _, _, _ = env.step(env.action_space.sample())
    img = env.render()
    stacked_images.append(transform_to_grayscale_and_normalize(img))
    frames.append(img)

# env.reset()
steps = 3

while True:
    X = torch.stack(stacked_images).unsqueeze(0)

    action = model(X).max(1)[1].view(1, 1).item()

    # take action
    _, _, terminated, truncated, _ = env.step(action)

    # update image stack with new state
    img = env.render()
    stacked_images = stacked_images[1:]
    stacked_images.append(transform_to_grayscale_and_normalize(img))

    frames.append(img)

    steps += 1

    if terminated or steps >= 10000:# or truncated:
        print(f"finished in {steps} steps")
        imageio.mimsave(f'video/replay.gif', frames, fps=40)
        break


