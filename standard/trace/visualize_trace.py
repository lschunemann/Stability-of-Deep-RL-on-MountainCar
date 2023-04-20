import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import seaborn as sns
from standard.helper import get_closest_in_grid, initialize_grids, initialize_state_dict

trace = []
actions = []

# load trace and actions into list
with open('trace_eval.txt') as f:
    for line in f.readlines():
        x, v, a = line.split(',')
        trace.append((float(x), float(v)))
        actions.append(int(a))

# discretize trace
trace_discrete = []
grid_x, grid_v = initialize_grids()
state_to_q = initialize_state_dict()
for trac in trace:
    # state = state_to_q[get_closest_in_grid(np.asarray(trac), grid_x, grid_v)]
    x, v = get_closest_in_grid(np.asarray(trac), grid_x, grid_v)
    state = [grid_x.index(x)+0.5, grid_v.index(v)+0.5]
    trace_discrete.append(state)

# load v values plot
plot = img.imread('../plots/v_values_rand_decay.png')

# plot trace onto plot
fig, ax = plt.subplots()
# ax.imshow(plot, extent=[0, 20, 0, 20])
with open('../plots/v_values.pkl', 'rb') as fid:
    ax = pickle.load(fid)
ax.plot(*zip(*trace_discrete[::1]))

# plot starting point
sns.lineplot(x=[trace_discrete[0][0]], y=[trace_discrete[0][1]], marker='o', markersize=25, markeredgecolor='green', color='green', markeredgewidth=2)
# plot finish point
sns.lineplot(x=[trace_discrete[-1][0]+1], y=[trace_discrete[-1][1]], marker='o', markersize=25, markeredgecolor='red', color='red', markeredgewidth=2)

ax.set_xticklabels(grid_x)
ax.set_yticklabels(grid_v)
plt.savefig('trace.png')
plt.close()

