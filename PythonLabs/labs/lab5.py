import numpy as np
from matplotlib import pyplot as plt
from pylab import rcParams, datetime
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D  # let Axes3D to register itself

from lorenz import modelling

rcParams['lines.linewidth'] = 1.0

S = 10
b = 8.0 / 3.0
r = 28

INITIAL = (1, 1, 1)
DT = 1e-2
TIME = 250

T_INIT = 0.5
P_INIT = (-1.40217687878, -16.230016085, 34.9398313536)

# Value depends on the dt and the initial point
# This value was received for 1e-2 and P_INIT
# Thus it corresponds to the time 7 * 0.01 = 0.07
TAU_IDX = 7


def plot(x, y, title, x_label, y_label, **kwargs):
    return plot_on(plt.figure(title).gca(), x, y, title, x_label, y_label, **kwargs)


def plot_3d(x, y, z, title, x_label, y_label, z_label):
    axes = plt.figure(title).gca(projection='3d')
    axes.plot(x, y, z)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_zlabel(z_label)
    axes.set_title(title)

    return axes


def plot_on(axes, x, y, title, x_label, y_label, **kwargs):
    axes.plot(x, y, **kwargs)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_title(title)

    return axes


# Family of C(r) correlation functions

# def C_1d(x, r):
#     N = len(x)
#     result = 0.0
#
#     for j in xrange(N - 1):
#         dx = x[j] - x[j + 1:]
#         result += np.sum((r - np.abs(dx)) > 0)
#
#     return (2 * result) / (N ** 2)
#
#
# def C_2d(x, y, r):
#     N = len(x)
#     result = 0.0
#
#     for j in xrange(N - 1):
#         dx = x[j] - x[j + 1:]
#         dy = y[j] - y[j + 1:]
#
#         result += np.sum((r - np.sqrt(dx ** 2 + dy ** 2)) > 0)
#
#     return (2 * result) / (N ** 2)


def C_1d_batched(x, r_values):
    N = len(x)
    result = np.full(len(r_values), 0.0)

    for j in xrange(N - 1):
        dx = x[j] - x[j + 1:]
        distances = np.abs(dx)

        for i in xrange(len(r_values)):
            result[i] += np.sum((r_values[i] - distances) > 0)

    return (2 * result) / (N ** 2)


def C_2d_batched(x, y, r_values):
    N = len(x)
    result = np.full(len(r_values), 0.0)

    for j in xrange(N - 1):
        dx = x[j] - x[j + 1:]
        dy = y[j] - y[j + 1:]
        distances = np.sqrt(dx ** 2 + dy ** 2)

        for i in xrange(len(r_values)):
            result[i] += np.sum((r_values[i] - distances) > 0)

    return (2 * result) / (N ** 2)


def C_3d_batched(x, y, z, r_values):
    N = len(x)
    result = np.full(len(r_values), 0.0)

    for j in xrange(N - 1):
        dx = x[j] - x[j + 1:]
        dy = y[j] - y[j + 1:]
        dz = z[j] - z[j + 1:]

        distances = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        for i in xrange(len(r_values)):
            result[i] += np.sum((r_values[i] - distances) > 0)

    return (2 * result) / (N ** 2)


def R(seq, t):
    if t == 0:
        return (seq * seq).mean()
    else:
        return (seq[t:] * seq[:-t]).mean()


def R_correlations(values):
    n_max = len(values)

    # Calculate auto-correlation
    t_values = range(0, n_max - 1)
    corr_values = np.array([R(values, t) for t in t_values])

    # Find the first local minimum
    min_t = np.where(corr_values[1:] - corr_values[:-1] > 0)[0][0]

    # Auto-correlation plot
    plot(t_values, corr_values, 'R(N)', 'N', 'R(N)').axvline(x=min_t, color='r', linewidth=0.7)

    return min_t


def select_tau(values):
    # Print plots and find the first local minimum
    local_min = R_correlations(values)

    print 'Tau = {}'.format(local_min)

    # Reconstruction in the most likely region
    fig = plt.figure('Reconstructions')

    shifts = np.arange(7, 11, 1)
    for i, shift in enumerate(shifts):
        x_val = values[:-shift]
        y_val = values[shift:]

        fig.add_subplot(2, 2, i + 1).plot(x_val, y_val)


def angle(x, y):
    tg_value = (y[-1] - y[0]) / (x[-1] - x[0])
    return np.arctan(tg_value) * 180.0 / np.pi


def C_plot(x, y, z, x_, y_, z_):
    # Compute C(r) values
    r_values = range(1, 11)

    c_recons = np.empty([3, 10])

    print '{}: Start computions'.format(datetime.datetime.now())

    c_recons[0] = C_1d_batched(x_, r_values)
    print '{}: 1D for reconstructed done'.format(datetime.datetime.now())

    c_recons[1] = C_2d_batched(x_, y_, r_values)
    print '{}: 2D for reconstructed done'.format(datetime.datetime.now())

    c_recons[2] = C_3d_batched(x_, y_, z_, r_values)
    print '{}: 3D for reconstructed done'.format(datetime.datetime.now())

    c_real = C_3d_batched(x, y, z, r_values)
    print '{}: 3D for real done'.format(datetime.datetime.now())

    r_logs = np.log(r_values)

    # Print plots
    axes = plt.figure(num='ln C(r) of ln r').gca()
    axes.set_xlabel('ln r')
    axes.set_ylabel('ln C(r)')

    for i, c in enumerate(c_recons):
        c_logs = np.log(c)
        ang = angle(r_logs, c_logs)

        axes.plot(r_logs, np.log(c), label='D = %d, angle = %.3f' % (i + 1, ang))

    c_logs = np.log(c_real)
    ang = angle(r_logs, c_logs)

    axes.plot(r_logs, c_logs, label='Real 3D values, angle = %.3f' % ang)


def lab_task():
    # Looking for the beginning of the attractor
    # time = TIME
    # initial = INITIAL
    # plot(time_values(time, DT), x, 'X(T)', 'T', 'X')

    time = 250 - T_INIT
    initial = P_INIT

    x, y, z = modelling(S, b, r, initial, time, DT)
    plot_3d(x, y, z, 'Real attractor', 'X', 'Y', 'Z')

    # select_tau(x)

    tau = TAU_IDX * DT
    n_max = len(x)
    n = min(10000, n_max - 2 * TAU_IDX)

    x_ = x[:n]
    y_ = x[TAU_IDX:n + TAU_IDX]
    z_ = x[2 * TAU_IDX: n + 2 * TAU_IDX]

    # Reconstructed attractor 2D
    plot(x_, y_, '2D, tau = %.3f' % tau, 'X', 'Y')

    # Reconstructed attractor 3D
    plot_3d(x_, y_, z_, '3D, tau = %.3f' % tau, 'X', 'Y', 'Z')

    C_plot(x[:n], y[:n], z[:n], x_, y_, z_)

    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    lab_task()
