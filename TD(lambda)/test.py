import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
# from train import getActiveTiles, epsilon_greedy
from TileCoding import *

env = gym.make('MountainCar-v0', render_mode="rgb_array")

# for i in range(3):
#     w = (i+1) * ((1 / 20) / 3) * np.asarray([1, 3])
#     print(w)
# maxsize = 2048
# numtilings = 8
# epsilon = 0.05
# w = np.zeros(maxsize)
# state = env.reset()
# action = epsilon_greedy(env, w, epsilon)
# x_bound = [-1.2, 0.6]
# v_bound = [-0.07, 0.07]
# print(state)
# print(state[0] * (numtilings / x_bound[1] - x_bound[0]))


# def normalize_state(state, numtilings):
#     x = state[0] * (numtilings / (x_bound[1] - x_bound[0]))
#     v = state[1] * (numtilings / (v_bound[1] - v_bound[0]))
#     return [x, v]
#
#
# x_bound = [-1.2, 0.6]
# v_bound = [-0.07, 0.07]
#
# maxsize = 2048
# w = np.zeros(maxsize)
# actions = [-1, 0, 1]
# hash_table = IHT(maxsize)
# numtilings = 8
# z = np.zeros(w.shape)
#
# state = env.reset()[0]
#
# for action in actions:
#     print(tiles(hash_table, numtilings, normalize_state(state, numtilings), [action]))
#     # new_state, reward, terminated, truncated, info = env.step(action)
#     # print(tiles(hash_table, numtilings, normalize_state(new_state, numtilings), [action]))
#
# for i in range(100):
#     action = env.action_space.sample()
#     print(tiles(hash_table, numtilings, normalize_state(state, numtilings), [action]))

# import torchvision
# import torch
# images = []
# state = env.reset()[0]
# for i in range(4):
#     img = env.render()
#     img = np.transpose(img, (2, 0, 1))
#     img = torch.from_numpy(img)
#     img = torchvision.transforms.functional.rgb_to_grayscale(img, 1) / 255
#     images.append(torch.squeeze(img))
#     state = env.step(env.action_space.sample())
#
# X = torch.stack(images)
#
# print(X.max(1)[1])
#
#
# def conv_out_shape(shape, kernel, stride, padding):
#     h = ((shape[0] + 2 * padding[0] - (kernel[0] - 1) - 1) / stride[0]) + 1
#     w = ((shape[1] + 2 * padding[1] - (kernel[1] - 1) - 1) / stride[1]) + 1
#     return h, w
#
# def pool_out_shape(shape, kernel, stride):
#     h = ((shape[0] - (kernel[0] - 1) - 1) / stride[0]) + 1
#     w = ((shape[1] - (kernel[1] - 1) - 1) / stride[1]) + 1
#     return h, w

# print(conv_out_shape((np.asarray(conv_out_shape((np.asarray(conv_out_shape((400,600), (8, 8), (2, 2), (4, 6))) / 2), (5,5), (2,2), (2,2))) / 2), (2,2), (2,2), (0,0)))
# print(np.asarray(conv_out_shape((400, 600), (8, 8), (2, 2), (1, 1))))
# print(np.asarray(pool_out_shape((198, 298), (2, 2), (2, 2))))
# print(np.asarray(conv_out_shape((99, 149), (5, 5), (2, 2), (0, 1))))
# print(np.asarray(pool_out_shape((48, 74), (2, 2), (2, 2))))
# print(np.asarray(conv_out_shape((24, 37), (5, 5), (1, 2), (0, 1))))
# print(np.asarray(pool_out_shape((20, 18), (2, 2), (2, 2))))
# print(np.asarray(conv_out_shape((10, 9), (5, 5), (1, 2), (0, 1))))
# print(np.asarray(pool_out_shape((6, 4), (2, 2), (2, 2))))

# conv sizes:
# [ 99. 149.]
# [24. 37.]
# [10.  9.]
# [3. 2.]

# import random
# l = []
# for i in range(100):
#     l.append([torch.rand(10), random.choice([True, False])])
# print()
# l = np.asarray(l, dtype=object)
# print(l[:,0])

# plt.imshow(img.permute(1, 2, 0))
# plt.show()

# print(np.asarray(conv_out_shape((84, 84), (8, 8), (4, 4), (0, 0))))
# print(np.asarray(conv_out_shape((20, 20), (4, 4), (2, 2), (0, 0))))
# print(np.asarray(conv_out_shape((9, 9), (3, 3), (1, 1), (0, 0))))

# conv sizes:
# [ 20. 20.]
# [9. 9.]
# [7. 7.]
#
# print(np.asarray(conv_out_shape((80, 120), (8, 8), (4, 4), (0, 0))))
# print(np.asarray(conv_out_shape((19, 29), (5, 5), (2, 2), (0, 0))))
# print(np.asarray(conv_out_shape((8, 13), (3, 3), (1, 1), (0, 0))))

# conv sizes:
# [ 19. 29.]
# [8. 13.]
# [6. 11.]

# linear function to decay epsilon
# -((0.9-0.1)/1000000) * x + 0.9
def normalize_state(state, numtilings, x_bound, v_bound):
    x = state[0] * (numtilings / (x_bound[1] - x_bound[0]))
    v = state[1] * (numtilings / (v_bound[1] - v_bound[0]))
    return [x, v]

def scale_state(state, x_bound, v_bound):
    scaleFactorX = 10.0 / (x_bound[1] - x_bound[0])
    scaleFactorV = 10.0 / (v_bound[1] - v_bound[0])
    return [state[0] * scaleFactorX, state[1] * scaleFactorV]

x_bound = [-1.2, 0.6]
v_bound = [-0.07, 0.07]

from TileCoding import IHT, tiles

iht = IHT(2000)

# print('state -> indices')
# for x,v in zip(np.arange(-1.2, 0.6, 0.1), np.arange(-0.07, 0.07, 0.008)):
#     state = normalize_state([x,v], 8, x_bound, v_bound)
#     indices = tiles(iht, numtilings=8, floats=state)
#     print('{0:.2f}'.format(x), ', ' '{0:.2f}'.format(v), ' -> ', indices)
print('state -> indices')
for x in np.arange(-1.2, 0.6, 0.1):
    for v in np.arange(-0.07, 0.07, 0.008):
        state = scale_state([x,v], x_bound , v_bound)
        indices = tiles(iht, numtilings=8, floats=state)
        print('{0:.2f}'.format(x), ', ' '{0:.2f}'.format(v), ' -> ', indices)
