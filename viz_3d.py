import numpy as np
import matplotlib.pyplot as plt

q = np.loadtxt('data/q_car.txt')
chosen = np.max(q, axis=1)
chosen = chosen.reshape((20, 20))

ny, nx = chosen.shape
x = np.linspace(-1.2, 0.6, nx)
y = np.linspace(-0.7, 0.7, ny)
xv, yv = np.meshgrid(x, y)

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(projection='3d')

ax.plot_wireframe(xv, yv, chosen)
ax.set_xlabel('Position')
ax.set_ylabel('Velocity')
ax.set_zlabel('V value')
plt.savefig('plots/q_car_3d.png')
