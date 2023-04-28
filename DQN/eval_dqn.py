import gymnasium as gym
import torch
from helper_DQN import transform_to_grayscale_and_normalize
from models import DQN

# Evaluation Hyperparameters
n_eval_episodes = 100
max_steps = 10000

# load model
model = DQN(3)
model.load_state_dict(torch.load("data/DQN.pth"))
model.eval()

# initialize environment
env = gym.make("MountainCar-v0", render_mode='rgb_array')


for episode in range(n_eval_episodes):
    stacked_images = []
    img = env.render()
    stacked_images.append(transform_to_grayscale_and_normalize(img))

    for i in range(3):
        _, _, _, _, _ = env.step(env.action_space.sample())
        img = env.render()
        stacked_images.append(transform_to_grayscale_and_normalize(img))

    env.reset()
    steps = 0

    for i in range(max_steps):
        X = torch.stack(stacked_images)

        action = model(X).max(1)[1].view(1, 1).item()

        # take action
        _, _, terminated, _, _ = env.step(action)

        # update image stack with new state
        img = env.render()
        stacked_images = stacked_images[1:]
        stacked_images.append(transform_to_grayscale_and_normalize(img))

        steps += 1

        if terminated:
            print(f"finished in {steps} steps")
            break