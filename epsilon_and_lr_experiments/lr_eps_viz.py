import matplotlib.pyplot as plt
import matplotlib.image as img
import os

actions = []
rewards = []
steps = []
v_values = []


# load images into lists
i = 0
for plot in os.listdir("plots"):
    if i < 25:
        actions.append(img.imread('plots/'+plot))
        i += 1
        continue
    elif i < 50:
        rewards.append(img.imread('plots/'+plot))
        i += 1
        continue
    elif i < 75:
        steps.append(img.imread('plots/'+plot))
        i += 1
        continue
    else:
        v_values.append(img.imread('plots/'+plot))
        i += 1


fig_size = 50
font_x_y = 36
font_title = 50
# Create 5 x 5 plot for best actions
num_subplots = 5
fig, axs = plt.subplots(nrows=num_subplots, ncols=num_subplots, figsize=(fig_size, fig_size))

for i in range(num_subplots):
    for j in range(num_subplots):
        axs[i, j].imshow(actions[i*num_subplots+j])
        axs[i, j].get_xaxis().set_visible(False)
        axs[i, j].get_yaxis().set_visible(False)

fig.supxlabel('Learning rates\n0.001, 0.005, 0.01, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 2', fontsize=font_x_y)
fig.supylabel('Gamma\n0, 0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.99, 1', fontsize=font_x_y)
fig.suptitle('Learned best actions based on different values for gamma and lr', fontsize=font_title)
plt.savefig('best_actions_5x5.png')
plt.close()


# Create 5 x 5 plot for average rewards
fig, axs = plt.subplots(nrows=num_subplots, ncols=num_subplots, figsize=(fig_size, fig_size))

for i in range(num_subplots):
    for j in range(num_subplots):
        axs[i, j].imshow(rewards[i*num_subplots+j])
        axs[i, j].get_xaxis().set_visible(False)
        axs[i, j].get_yaxis().set_visible(False)

fig.supxlabel('Learning rates\n0.001, 0.005, 0.01, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 2', fontsize=font_x_y)
fig.supylabel('Gamma\n0, 0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.99, 1', fontsize=font_x_y)
fig.suptitle('Average rewards during learning based on different values for gamma and lr', fontsize=font_title)
plt.savefig('avg_rewards_5x5.png')
plt.close()


# Create 5 x 5 plot for total steps
fig, axs = plt.subplots(nrows=num_subplots, ncols=num_subplots, figsize=(fig_size, fig_size))

for i in range(num_subplots):
    for j in range(num_subplots):
        axs[i, j].imshow(steps[i*num_subplots+j])
        axs[i, j].get_xaxis().set_visible(False)
        axs[i, j].get_yaxis().set_visible(False)

fig.supxlabel('Learning rates\n0.001, 0.005, 0.01, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 2', fontsize=font_x_y)
fig.supylabel('Gamma\n0, 0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.99, 1', fontsize=font_x_y)
fig.suptitle('Total steps needed per Episode based on different values for gamma and lr', fontsize=font_title)
plt.savefig('total_steps_5x5.png')
plt.close()


# Create 10 x 10 plot for v_values
fig, axs = plt.subplots(nrows=num_subplots, ncols=num_subplots, figsize=(fig_size, fig_size))

for i in range(num_subplots):
    for j in range(num_subplots):
        axs[i, j].imshow(v_values[i*num_subplots+j])
        axs[i, j].get_xaxis().set_visible(False)
        axs[i, j].get_yaxis().set_visible(False)

fig.supxlabel('Learning rates\n0.001, 0.005, 0.01, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 2', fontsize=font_x_y)
fig.supylabel('Gamma\n0, 0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.99, 1', fontsize=font_x_y)
fig.suptitle('Learned V values based on different values for gamma and lr', fontsize=font_title)
plt.savefig('v_values_5x5.png')
plt.close()
