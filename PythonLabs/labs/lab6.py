import numpy as np
from matplotlib import pyplot as plt

from plotting import plot_on

T = 50.
dt = 1e-3
N = int(T / dt)
TIME = np.linspace(0, T, N)

A = 0.05
B = 0.26
C = -60
D = 0

K1 = 0.04
K2 = 5
K3 = 140
V_thr = 30

V_0 = -70
U_0 = -20

I_critical = ((K2 - B) ** 2) / (4 * K1) - K3
V_critical = (B - K2) / (2 * K1)
U_critical = B * V_critical
lambdas = np.array([0, B - A])

e = np.array([1, B - lambdas[1]])  # B - lambda = B - (B - A) = A
p = np.array([V_critical, U_critical])

factor = 2

p1 = p + factor * e
p2 = p - factor * e


def f(v):
    return K1 * (v ** 2) + K2 * v + K3


def Vt(v, u, I):
    return f(v) - u + I


def Ut(v, u):
    return A * (B * v - u)


def get_curve(I, init):
    v = np.empty(N)
    u = np.empty(N)
    v[0], u[0] = init
    count = 0

    for i in range(N - 1):
        v[i + 1] = v[i] + dt * Vt(v[i], u[i], I)
        u[i + 1] = u[i] + dt * Ut(v[i], u[i])

        if v[i] >= V_thr:
            v[i + 1] = C
            u[i + 1] = u[i] + D
            count += 1

    return v, u, count


def neuron_plot(I, init, plot_uv=False):
    v, u, count = get_curve(I, init)

    fig = plt.figure('Neuron plots, I = {}, v(0) = {}, u(0) = {}'.format(I, init[0], init[1]))

    grid = 130 if plot_uv else 120

    plot_on(fig.add_subplot(grid + 1), TIME, v, 'v(t)', 't', 'v')
    plot_on(fig.add_subplot(grid + 2), TIME, u, 'u(t)', 't', 'u')

    if plot_uv:
        plot_on(fig.add_subplot(grid + 3), u, v, 'u(v)', 'u', 'v')


def task_0_print_values():
    print 'I_cr = %.4f, (V_cr, U_cr) = (%.4f, %.4f)' % (I_critical, U_critical, V_critical)
    print 'Lambda values: %s' % lambdas
    print 'e = %s, p = %s' % (e, p)


def task_2_u_v_plots():
    neuron_plot(I_critical, p1, True)
    neuron_plot(I_critical, p2, True)


def task_3_different_I():
    init = [V_0, U_0]

    neuron_plot(0, init)
    neuron_plot(I_critical / 2.0, init)
    neuron_plot(I_critical * 2.0, init)


def task_5_peak_freqs():
    init = [V_0, U_0]

    I_values = [I_critical * n for n in range(1, 11)]
    count_values = [get_curve(I, init)[2] for I in I_values]

    fig = plt.figure('Peaks frequency, time = {}'.format(TIME))
    plot_on(fig.gca(), I_values, count_values, 'N(I)', 'I', 'N')


if __name__ == '__main__':
    task_0_print_values()
    task_2_u_v_plots()
    task_3_different_I()
    # I didn't understand task 4 :(
    task_5_peak_freqs()

    plt.show()
