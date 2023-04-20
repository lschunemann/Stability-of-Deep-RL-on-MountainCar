import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

td1 = np.loadtxt('data/steps_lr_0.1,lambda_0.2')
td2 = np.loadtxt('data/steps_lr_0.1,lambda_0.4')
td3 = np.loadtxt('data/steps_lr_0.1,lambda_0.5')
td4 = np.loadtxt('data/steps_lr_0.1,lambda_0.7')
td5 = np.loadtxt('data/steps_lr_0.1,lambda_0.9')
td6 = np.loadtxt('data/steps_lr_0.2,lambda_0.2')
td7 = np.loadtxt('data/steps_lr_0.2,lambda_0.4')
td8 = np.loadtxt('data/steps_lr_0.2,lambda_0.5')
td9 = np.loadtxt('data/steps_lr_0.2,lambda_0.7')
td10 = np.loadtxt('data/steps_lr_0.2,lambda_0.9')
td11 = np.loadtxt('data/steps_lr_0.3,lambda_0.2')
td12 = np.loadtxt('data/steps_lr_0.3,lambda_0.4')
td13 = np.loadtxt('data/steps_lr_0.3,lambda_0.5')
td14 = np.loadtxt('data/steps_lr_0.3,lambda_0.7')
td15 = np.loadtxt('data/steps_lr_0.3,lambda_0.9')
td16 = np.loadtxt('data/steps_lr_0.4,lambda_0.2')
td17 = np.loadtxt('data/steps_lr_0.4,lambda_0.4')
td18 = np.loadtxt('data/steps_lr_0.4,lambda_0.5')
td19 = np.loadtxt('data/steps_lr_0.4,lambda_0.7')
td20 = np.loadtxt('data/steps_lr_0.4,lambda_0.9')
td21 = np.loadtxt('data/steps_lr_0.5,lambda_0.2')
td22 = np.loadtxt('data/steps_lr_0.5,lambda_0.4')
td23 = np.loadtxt('data/steps_lr_0.5,lambda_0.5')
td24 = np.loadtxt('data/steps_lr_0.5,lambda_0.7')
td25 = np.loadtxt('data/steps_lr_0.5,lambda_0.9')

td_list = [td1, td2, td3, td4, td5, td6, td7, td8, td9, td10, td11, td12, td13, td14, td15, td16, td17, td18, td19,
           td20, td21, td22, td23, td24, td25]

time_steps = 50

labels = []

learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
lmbdas = [0.2, 0.4, 0.5, 0.7, 0.9]

for lr in learning_rates:
    for lmbda in lmbdas:
        labels.append(f'lr: {lr} - lambda: {lmbda}')

# td_1 = td_list[:5]
# td_2 = td_list[5:10]
# td_3 = td_list[10:15]
# td_4 = td_list[15:20]
# td_5 = td_list[20:]
# tds = [td_1, td_2, td_3, td_4, td_5]
#
# labels = [f'lambda: {ld}' for ld in lmbdas]
#
# for td_l, learning_rate in zip(tds, learning_rates):
#     plt.figure(figsize=(10, 7))
#     for index, i in enumerate([td[:time_steps] for td in td_l]):
#         plt.plot(range(len(i)), i, label=labels[index])
#     plt.legend()
#     plt.savefig(f'plots/compare_TD_lambda_lr_{learning_rate}.png')
#     plt.close()
#
#
# time_steps = 25
#
# td_09 = td_list[4::5]
#
# labels = [f'step size: {lr}' for lr in learning_rates]
#
# plt.figure(figsize=(10, 7))
# for index, i in enumerate([td[:time_steps] for td in td_09]):
#     plt.plot(range(len(i)), i, label=labels[index])
# plt.legend()
# plt.savefig(f'plots/compare_TD_stepsizes_lambda_0.9.png')
# plt.close()


plot1 = img.imread('plots/compare_TD_lambda_lr_0.1.png')
plot2 = img.imread('plots/compare_TD_lambda_lr_0.2.png')
plot3 = img.imread('plots/compare_TD_lambda_lr_0.3.png')
plot4 = img.imread('plots/compare_TD_lambda_lr_0.4.png')
plot5 = img.imread('plots/compare_TD_lambda_lr_0.5.png')
plots_list = [plot1, plot2, plot3, plot4, plot5]

# labels = [f'lambda: {ld}' for ld in lmbdas]

fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(18, 3))

for j, plot in enumerate(plots_list):
    axs[j].imshow(plot)
    axs[j].get_xaxis().set_visible(False)
    axs[j].get_yaxis().set_visible(False)

fig.suptitle('Best Lambda values for different step sizes', fontsize=15)
fig.supxlabel('Step sizes: 0.1, 0.2, 0.3, 0.4, 0.5', fontsize=10)
plt.savefig('plots/TD_lambda_lr_lmbda_compare.png')
plt.close()
