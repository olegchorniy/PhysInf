from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from memristor import Memristor

D = 10e-9  # 10 nm
MU = 1e-10 * 1e-4  # TODO?
NU0 = 100  # Hz
OMEGA_0 = NU0
PERIODS = 5

U0 = 1
W0 = D / 10.0


# -------------------------------------------------------------
# --------------- U functions and their integrals -------------
# -------------------------------------------------------------

def U_sin(u0, nu, t):
    return u0 * np.sin(nu * t)


def Phi_sin(u0, nu, t):
    return u0 * (1 - np.cos(nu * t)) / nu


def U_sin_sqr(u0, nu, t):
    return u0 * (np.sin(nu * t) ** 2)


def Phi_sin_sqr(u0, nu, t):
    return u0 * (2 * nu * t - np.sin(2 * nu * t)) / (4 * nu)


# ------------------------------------------------------------
# ----------------------- Main lab tasks ---------------------
# ------------------------------------------------------------


def plots_for(U, Phi):
    R_ON = 1.0
    N = 1000

    for R_OFF in R_ON * np.array([160.0, 380.0]):
        m = Memristor(D, MU, NU0, W0, R_ON, R_OFF, U, Phi)

        m.task1(plt, N, PERIODS)
        m.task2(plt, N, PERIODS)
        m.task3(plt, N, PERIODS)


def task_1():
    U = partial(U_sin, U0, NU0)
    Phi = partial(Phi_sin, U0, NU0)

    plots_for(U, Phi)


def task_2():
    U = partial(U_sin_sqr, U0, NU0)
    Phi = partial(Phi_sin_sqr, U0, NU0)

    plots_for(U, Phi)


def task_3():
    R_ON = 1.0
    N = 1000
    w0 = 0

    for u0 in [1.0, 2.0, 4.0]:
        U = partial(U_sin, u0, NU0)
        Phi = partial(Phi_sin, u0, NU0)

        for R_OFF in R_ON * np.array([50.0, 125.0]):
            m = Memristor(D, MU, NU0, w0, R_ON, R_OFF, U, Phi)

            m.task1(plt, N, PERIODS)


def task_4():
    R_ON = 1.0
    N = 1000
    w0 = 0

    for nu0 in [1.0, 10.0, 100.0, 1000.0]:
        U = partial(U_sin, U0, nu0)
        Phi = partial(Phi_sin, U0, nu0)

        for R_OFF in R_ON * np.array([50.0, 125.0]):
            m = Memristor(D, MU, NU0, w0, R_ON, R_OFF, U, Phi)

            m.task1(plt, N, PERIODS)


if __name__ == '__main__':
    # task_1()
    # task_2()
    # task_3()
    task_4()
    plt.show()
