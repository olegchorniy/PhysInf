import matplotlib.pyplot as plt
import numpy as np
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D  # let Axes3D to register itself

S = 10
B = 8.0 / 3.0
R = 28

# Critical R
R_CRITICAL = S * (S + B + 3.0) / (S - B - 1)

# modelling related constants
DEFAULT_TIME = 50
STEP = 1e-2
INITIAL = np.array([2, -1, 0])


# Plotting helpers

def three_dim_plot(axes, title, points, labels):
    axes.plot(points['x'], points['y'], points['z'])
    axes.set_xlabel(labels['x'])
    axes.set_ylabel(labels['y'])
    axes.set_zlabel(labels['z'])
    axes.set_title(title)


def two_dim_plot(axes, title, points, labels):
    axes.plot(points['x'], points['y'])
    axes.set_xlabel(labels['x'])
    axes.set_ylabel(labels['y'])
    axes.set_title(title)


class Lorenz(object):
    def __init__(self, sigma, b, r):
        self._s = sigma
        self._b = b
        self._r = r

    def derivative(self, state):
        return np.array(self.derivative_core(*state))

    def derivative_core(self, x, y, z):
        dx_dt = self._s * (y - x)
        dy_dt = x * (self._r - z) - y
        dz_dt = x * y - self._b * z

        return dx_dt, dy_dt, dz_dt


def modelling(r, modelling_time=DEFAULT_TIME):
    lorenz = Lorenz(S, B, r)
    num_points = int(modelling_time / STEP)

    trace = np.empty([num_points, 3])
    trace[0] = INITIAL

    for i in xrange(1, num_points):
        trace[i] = trace[i - 1] + STEP * lorenz.derivative(trace[i - 1])

    return trace.T


def attractor_for_various_r():
    time = DEFAULT_TIME
    fig = plt.figure("Lorenz Attractor, T = {}".format(time))

    r_midpoint = 1 + (R_CRITICAL - 1) / 2.0

    for i, r in enumerate([0.5, r_midpoint, R_CRITICAL + 1]):
        points = modelling(r, time)

        axes = fig.add_subplot(220 + i + 1, projection='3d')
        title = "R = {}".format(r)

        three_dim_plot(
            axes, title,
            {'x': points[0, :], 'y': points[1, :], 'z': points[2, :]},
            {'x': "X", 'y': "Y", 'z': "Z"}
        )


def plot_attractor_with_projections(r, time=DEFAULT_TIME):
    points = modelling(r, time)

    fig = plt.figure("Attractor, R = {}, T = {}".format(r, time))

    # Main plot
    three_dim_plot(
        fig.add_subplot(131, projection='3d'),
        "3D view",
        {'x': points[0, :], 'y': points[1, :], 'z': points[2, :]},
        {'x': "X", 'y': "Y", 'z': "Z"}
    )

    # XY-projection
    two_dim_plot(
        fig.add_subplot(132),
        "XY-projection",
        {'x': points[0, :], 'y': points[1, :]},
        {'x': "X", 'y': "Y"}
    )

    # XZ-projection
    two_dim_plot(
        fig.add_subplot(133),
        "XZ-projection",
        {'x': points[0, :], 'y': points[2, :]},
        {'x': "X", 'y': "Z"}
    )


if __name__ == '__main__':
    attractor_for_various_r()
    plot_attractor_with_projections(R_CRITICAL + 2)

    plt.show()
