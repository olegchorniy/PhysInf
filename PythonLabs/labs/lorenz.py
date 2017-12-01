import numpy as np

STEP = 1e-2


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


def modelling(sigma, b, r, initial_point, modelling_time, step=STEP):
    lorenz = Lorenz(sigma, b, r)
    num_points = int(modelling_time / step)

    trace = np.empty([num_points, 3])
    trace[0] = initial_point

    for i in xrange(1, num_points):
        trace[i] = trace[i - 1] + step * lorenz.derivative(trace[i - 1])

    return trace[:, 0], trace[:, 1], trace[:, 2]
