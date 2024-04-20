# import time

# import matplotlib.pyplot as plt
# import numpy as np

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# # Make the X, Y meshgrid.
# xs = np.linspace(-1, 1, 50)
# ys = np.linspace(-1, 1, 50)
# X, Y = np.meshgrid(xs, ys)

# # Set the z axis limits, so they aren't recalculated each frame.
# ax.set_zlim(-1, 1)

# # Begin plotting.
# wframe = None
# tstart = time.time()
# for phi in np.linspace(0, 180. / np.pi, 100):
#     # If a line collection is already remove it before drawing.
#     if wframe:
#         wframe.remove()
#     # Generate data.
#     Z = np.cos(2 * np.pi * X + phi) * (1 - np.hypot(X, Y))
#     # Plot the new wireframe and pause briefly before continuing.
#     wframe = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)
#     plt.pause(.001)

# print('Average FPS: %f' % (100 / (time.time() - tstart)))


import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Grab some example data and plot a basic wireframe.
X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

# Set the axis labels
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Rotate the axes and update
for angle in range(0, 360*4 + 1):
    # Normalize the angle to the range [-180, 180] for display
    angle_norm = (angle + 180) % 360 - 180

    # Cycle through a full rotation of elevation, then azimuth, roll, and all
    elev = azim = roll = 0
    if angle <= 360:
        elev = angle_norm
    elif angle <= 360*2:
        azim = angle_norm
    elif angle <= 360*3:
        roll = angle_norm
    else:
        elev = azim = roll = angle_norm

    # Update the axis view and title
    ax.view_init(elev, azim, roll)
    plt.title('Elevation: %d°, Azimuth: %d°, Roll: %d°' % (elev, azim, roll))

    plt.draw()
    plt.pause(.001)