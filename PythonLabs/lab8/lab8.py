from functools import partial

import numpy as np
from matplotlib import pyplot as plt
from pylab import rcParams

rcParams['lines.markersize'] = 3.0
rcParams['lines.linewidth'] = 1.0

DEFAULT_MU = 3.7
DEFAULT_N = 1000

# Experimentally determined values
EPS_SYNC = 0.17
LIM_VALUES = -37.0


# ------------------- Logistic related functions --------------

def logistic_mapping(x, mu):
    return mu * x * (1 - x)


def logistic_generator(x_initial, mu, n):
    if n <= 0:
        return

    yield x_initial

    x = x_initial
    for _ in xrange(n - 1):
        x = logistic_mapping(x, mu)
        yield x


def logistic_sequence(x_initial, mu, n=DEFAULT_N):
    return [x for x in logistic_generator(x_initial, mu, n)]


# ------------------- Synchronization ---------------------


def synchronized_values(x, y, mu, eps):
    f_x = logistic_mapping(x, mu)
    f_y = logistic_mapping(y, mu)

    return f_x - eps * (f_x - f_y), f_y + eps * (f_x - f_y)


def synchronized_transformed(x, y):
    return (x + y) / 2.0, (x - y) / 2.0


def synchronized_sequence(x_init, y_init, eps, mu=DEFAULT_MU, n=DEFAULT_N):
    x_seq, y_seq = np.empty(n), np.empty(n)
    x_seq[0] = x_init
    y_seq[0] = y_init

    x, y = x_init, y_init
    for i in xrange(1, n):
        x, y = synchronized_values(x, y, mu, eps)

        x_seq[i] = x
        y_seq[i] = y

    return x_seq, y_seq


# ------------------- Other functions ----------------------


def second_order_cycle_curve(x, mu=DEFAULT_MU):
    mu_2 = mu ** 2
    mu_3 = mu ** 3

    return mu_3 * (x ** 3) - (mu_3 + mu_2) * (x ** 2) + mu_2 * x


def sec_order_discriminant(mu):
    mu_2 = mu ** 2
    mu_3 = mu ** 3
    return (mu_2 + mu_3) ** 2 - 4 * mu_3 * (mu_2 - 1)


def sec_order_roots(mu):
    D = sec_order_discriminant(mu)
    D_root = D ** 0.5
    mu_3 = mu ** 3
    mu_2 = mu ** 2
    t = mu_3 + mu_2
    t2 = 2 * mu_3

    return [(t + D_root) / t2, (t - D_root) / t2]


# ------------------ Plotting helpers ---------------------

def two_dim_plot(axes, title, x_label, y_label, x, y):
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_title(title)
    axes.plot(x, y)


def scatter(axes, title, x_label, y_label, x, y):
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_title(title)
    axes.scatter(x, y)


# ------------------- Functions plotting --------------------

def plot_identity(x_from=0, x_to=1.0):
    plot_function('y = x', lambda x: x, x_from, x_to, 2)


def plot_zero(x_from=0.0, x_to=1.0):
    plot_function('y = 0', lambda x: 0, x_from, x_to, 2)


def plot_function(label, func, x_from, x_to, num=1000):
    x = np.linspace(x_from, x_to, num, True)
    y = map(func, x)

    plt.plot(x, y, label=label)


def plot_logistic(mu=DEFAULT_MU):
    label = 'Logistic, mu = {}'.format(mu)
    func = partial(logistic_mapping, mu=mu)

    plot_function(label, func, 0, 1.0, DEFAULT_N)


def plot_second_order_cycle(mu=DEFAULT_MU):
    title = 'Cycle 2, mu = {}'.format(mu)
    func = partial(second_order_cycle_curve, mu=mu)

    plot_function(title, func, 0.0, 1.0)


def plot_second_order_roots():
    plot_zero()
    plot_function('y = 1', lambda x: 1, 0.0, 4.0, 2)

    plot_function('Root 1', lambda mu: sec_order_roots(mu)[0], 0.00001, 4.0, 10000)
    plot_function('Root 2', lambda mu: sec_order_roots(mu)[1], 0.00001, 4.0, 10000)


def plot_sequences():
    mu = 3.7
    n = np.arange(1, DEFAULT_N + 1)
    fig = plt.figure('Logistic, mu = {}'.format(mu))

    for i, initial in enumerate([0.1, 0.3, 0.6, 0.9]):
        seq = logistic_sequence(initial, mu)
        axes = fig.add_subplot(411 + i)
        title = 'x(1) = %f' % initial

        two_dim_plot(axes, title, 'n', 'x(n)', n, seq)


def plot_next_of_prev():
    mu = 3.7
    n = 100
    fig = plt.figure('x(n+1) of x(n), mu = {}'.format(mu))

    for i, initial in enumerate([0.1, 0.3, 0.6, 0.9]):
        seq = logistic_sequence(initial, mu, n)
        axes = fig.add_subplot(221 + i)
        title = 'x(1) = %f' % initial

        two_dim_plot(axes, title, 'x(n)', 'x(n+1)', seq[0:n - 1], seq[1:n])


def plot_synchronized_values():
    x_initial = 0.05
    y_initial = 0.4
    n = DEFAULT_N
    n_values = range(1, n + 1)

    rows, cols = 5, 2

    xy_ax = plt.subplots(rows, cols, sharex='col', sharey='row', num='X, Y values')[1].flatten()
    uv_ax = plt.subplots(rows, cols, sharex='col', sharey='row', num='U, V values')[1].flatten()
    log_v_ax = plt.subplots(rows, cols, sharex='col', sharey='row', num='log |v(n)| values')[1].flatten()
    lim_ax = plt.subplots(rows, cols, sharex='col', sharey='row', num='lim log |v(n)| / n')[1].flatten()

    # for i, eps in enumerate([0.16, 0.17, 0.18, 0.19]):
    for i, eps in enumerate(np.linspace(0, 0.5, 11, False)[1:]):
        x, y = synchronized_sequence(x_initial, y_initial, eps, n=n)
        u, v = synchronized_transformed(x, y)
        log_values = np.log(np.abs(v))

        title = 'Eps = %.3f' % eps

        xy_ax[i].set_title(title)
        xy_ax[i].scatter(x, y)

        uv_ax[i].set_title(title)
        uv_ax[i].scatter(u, v)

        log_v_ax[i].set_title(title)
        log_v_ax[i].plot(n_values, log_values)

        lim_ax[i].set_title(title)
        lim_ax[i].plot(n_values, log_values / n_values)


if __name__ == '__main__':
    plot_next_of_prev()
    plot_sequences()
    plot_synchronized_values()

    plt.legend(loc=3)
    plt.show()
