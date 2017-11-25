# noinspection PyUnresolvedReferences
import matplotlib.pyplot as plt
import numpy as np

S = 10
B = 8.0 / 3.0
R = 28

# modelling related constants
TIME = [20, 50, 250]
STEP = 1e-2
INITIAL = np.array([2, -1, 0])


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


def critical_r():
    r1 = 1  # constant
    r2 = S * (S + B + 3.0) / (S - B - 1)

    return r1, r2


def modelling(modelling_time, r):
    lorenz = Lorenz(S, B, r)
    num_points = int(modelling_time / STEP)

    trace = np.empty([num_points, 3])
    trace[0] = INITIAL

    for i in xrange(1, num_points):
        trace[i] = trace[i - 1] + STEP * lorenz.derivative(trace[i - 1])

    return trace.T


def plot(title, plt_points):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(plt_points[0, :], plt_points[1, :], plt_points[2, :])
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(title)

    plt.show()


if __name__ == '__main__':
    r = critical_r()[1]
    points = modelling(TIME[2], r)

    print "Critical R = {}".format(r)
    print "Build plot from {} points".format(points)

    plot("Lorenz Attractor", points)
