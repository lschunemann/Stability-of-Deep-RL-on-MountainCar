import matplotlib.pyplot as plt
import matplotlib.image as img
import os


actions = []
v_values = []

# load images into lists
i = 0
for plot in os.listdir("plots"):
    if i < 9:
        actions.append(img.imread('plots/'+plot))
        i += 1
        continue
    else:
        v_values.append(img.imread('plots/'+plot))
        i += 1

# change order of lists
Y = [4,7,8,5,2,0,3,6,1]

actions = [x for _, x in sorted(zip(Y, actions))]
v_values = [x for _, x in sorted(zip(Y, v_values))]

# Create 1 x 9 plot for actions
fig, axs = plt.subplots(nrows=1, ncols=9, figsize=(32, 3))

for j in range(9):
    axs[j].imshow(actions[j])
    axs[j].get_xaxis().set_visible(False)
    axs[j].get_yaxis().set_visible(False)


# fig.supxlabel('best Actions at Episodes: 0, 4, 9, 25, 50, 100, 200, 500, 1000, 1999', fontsize=10)
# fig.supylabel('Gamma\n0, 0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.99, 1', fontsize=10)
fig.suptitle('Learned best actions based on different values for gamma and lr', fontsize=15)
plt.savefig('best_actions_1x9.png')
plt.close()


# Create 1 x 9 plot for v_values
fig, axs = plt.subplots(nrows=1, ncols=9, figsize=(32, 3))

for j in range(9):
    axs[j].imshow(v_values[j])
    axs[j].get_xaxis().set_visible(False)
    axs[j].get_yaxis().set_visible(False)

# fig.supxlabel('V Values at Episodes: 0, 4, 9, 25, 50, 100, 200, 500, 1000, 1999', fontsize=10)
# fig.supylabel('Gamma\n0, 0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.99, 1', fontsize=10)
fig.suptitle('Learned V values based on different values for gamma and lr', fontsize=15)
plt.savefig('v_values_1x9.png')
plt.close()
