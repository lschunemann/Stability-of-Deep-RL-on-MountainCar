import matplotlib.pyplot as plt
import matplotlib.image as img
import os

import numpy as np

actions = []
rewards = []
steps = []
v_values = []


# load images into lists
i = 0
for plot in os.listdir("plots"):
    if i < 100:
        actions.append(img.imread('plots/'+plot))
        i += 1
        continue
    elif i < 200:
        rewards.append(img.imread('plots/'+plot))
        i += 1
        continue
    elif i < 300:
        steps.append(img.imread('plots/'+plot))
        i += 1
        continue
    else:
        v_values.append(img.imread('plots/'+plot))
        i += 1

# Y1 = [i for i in range(80, 90)]
# Y2 = [i for i in range(80)]
# Y3 = [i for i in range(90, 100)]
# Y = np.concatenate((Y1, Y2, Y3))
# actions = [x for _, x in sorted(zip(Y, actions))]
# rewards = [x for _, x in sorted(zip(Y, rewards))]
# steps = [x for _, x in sorted(zip(Y, steps))]
# v_values = [x for _, x in sorted(zip(Y, v_values))]


fig_size = 50
font_x_y = 36
font_title = 50
# Create 10 x 10 plot for best actions
num_subplots = 10
fig, axs = plt.subplots(nrows=num_subplots, ncols=num_subplots, figsize=(fig_size, fig_size))

for i in range(10):
    for j in range(10):
        axs[i, j].imshow(actions[i*10+j])
        axs[i, j].get_xaxis().set_visible(False)
        axs[i, j].get_yaxis().set_visible(False)

fig.supxlabel('Learning rates\n0.001, 0.005, 0.01, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 2', fontsize=font_x_y)
fig.supylabel('Gamma\n1, 0.99, 0.95, 0.9, 0.85, 0.75, 0.5, 0.25, 0.1, 0', fontsize=font_x_y)
fig.suptitle('Learned best actions based on different values for gamma and lr', fontsize=font_title)
plt.savefig('best_actions_10x10')
plt.close()


# Create 10 x 10 plot for average rewards
fig, axs = plt.subplots(nrows=num_subplots, ncols=num_subplots, figsize=(fig_size, fig_size))

for i in range(10):
    for j in range(10):
        axs[i, j].imshow(rewards[i*10+j])
        axs[i, j].get_xaxis().set_visible(False)
        axs[i, j].get_yaxis().set_visible(False)

fig.supxlabel('Learning rates\n0.001, 0.005, 0.01, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 2', fontsize=font_x_y)
fig.supylabel('Gamma\n0, 0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.99, 1', fontsize=font_x_y)
fig.suptitle('Average rewards during learning based on different values for gamma and lr', fontsize=font_title)
plt.savefig('avg_rewards_10x10')
plt.close()


# Create 10 x 10 plot for total steps
fig, axs = plt.subplots(nrows=num_subplots, ncols=num_subplots, figsize=(fig_size, fig_size))

for i in range(10):
    for j in range(10):
        axs[i, j].imshow(steps[i*10+j])
        axs[i, j].get_xaxis().set_visible(False)
        axs[i, j].get_yaxis().set_visible(False)

fig.supxlabel('Learning rates\n0.001, 0.005, 0.01, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 2', fontsize=font_x_y)
fig.supylabel('Gamma\n0, 0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.99, 1', fontsize=font_x_y)
fig.suptitle('Total steps needed per Episode based on different values for gamma and lr', fontsize=font_title)
plt.savefig('total_steps_10x10')
plt.close()


# Create 10 x 10 plot for v_values
fig, axs = plt.subplots(nrows=num_subplots, ncols=num_subplots, figsize=(fig_size, fig_size))

for i in range(10):
    for j in range(10):
        axs[i, j].imshow(v_values[i*10+j])
        axs[i, j].get_xaxis().set_visible(False)
        axs[i, j].get_yaxis().set_visible(False)

fig.supxlabel('Learning rates\n0.001, 0.005, 0.01, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 2', fontsize=font_x_y)
fig.supylabel('Gamma\n0, 0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.99, 1', fontsize=font_x_y)
fig.suptitle('Learned V values based on different values for gamma and lr', fontsize=font_title)
plt.savefig('v_values_10x10')
plt.close()
