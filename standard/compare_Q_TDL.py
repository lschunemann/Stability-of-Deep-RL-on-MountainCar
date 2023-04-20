import numpy as np
import matplotlib.pyplot as plt

q_steps = np.loadtxt('data/total_steps.txt')
q_rand_steps = np.loadtxt('data/total_steps_rand_decay.txt')
td_l_steps = np.loadtxt('../TD(lambda)/data/steps.txt')

time_steps = 200

labels = ["Q learning", "Q learning w/ random init", "TD lambda"]
plt.figure(figsize=(10, 7))
for index, i in enumerate([q_steps[:time_steps], q_rand_steps[:time_steps], td_l_steps[:time_steps]]):
    plt.plot(range(len(i)), i, label=labels[index])
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Amount ot steps')
plt.savefig('plots/compare_methods.png')
plt.close()
