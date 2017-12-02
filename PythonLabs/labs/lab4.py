import matplotlib.pyplot as plt
import numpy as np

import lorenz
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
INITIAL = np.array([2, -1, 2])


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


def time_values(modelling_time=DEFAULT_TIME, step=STEP):
    num_points = int(modelling_time / step)
    return np.linspace(0, modelling_time, num_points, True)


def attractor_for_various_r(time=DEFAULT_TIME):
    fig = plt.figure("Lorenz Attractor, T = {}".format(time))

    r_midpoint = 1 + (R_CRITICAL - 1) / 2.0

    for i, r in enumerate([0.5, r_midpoint, R_CRITICAL + 1]):
        x, y, z = lorenz.modelling(S, B, r, INITIAL, time)

        axes = fig.add_subplot(220 + i + 1, projection='3d')
        title = "R = {}".format(r)

        three_dim_plot(
            axes, title,
            {'x': x, 'y': y, 'z': z},
            {'x': "X", 'y': "Y", 'z': "Z"}
        )


def plot_attractor_with_projections(r, time=DEFAULT_TIME):
    x, y, z = lorenz.modelling(S, B, r, INITIAL, time)

    fig = plt.figure("Attractor, R = {}, T = {}".format(r, time))

    # Main plot
    three_dim_plot(
        fig.add_subplot(131, projection='3d'),
        "3D view",
        {'x': x, 'y': y, 'z': z},
        {'x': "X", 'y': "Y", 'z': "Z"}
    )

    # XY-projection
    two_dim_plot(
        fig.add_subplot(132),
        "XY-projection",
        {'x': x, 'y': y},
        {'x': "X", 'y': "Y"}
    )

    # XZ-projection
    two_dim_plot(
        fig.add_subplot(133),
        "XZ-projection",
        {'x': x, 'y': z},
        {'x': "X", 'y': "Z"}
    )


def plot_volumes(r, time=DEFAULT_TIME):
    time_points = time_values(time)

    # Real volume values
    v_initial = INITIAL.prod()
    real_phase_volume = v_initial * np.exp(-(S + 1 + B) * time_points)

    # Modelled volume values
    x, y, z = lorenz.modelling(S, B, r, INITIAL, time)
    calculated_phase_volume = x * y * z

    # Plot on a separate plane
    fig = plt.figure('Phase volume')
    axes = fig.gca()

    axes.set_title('Phase volume')
    axes.set_xlabel('Time')
    axes.set_ylabel('Volume')

    axes.plot(time_points, calculated_phase_volume)
    axes.plot(time_points, real_phase_volume)


def different_initials(r, time=DEFAULT_TIME):
    initials = [
        [10, 10, 10],
        [10, -10, 4],
        [7, 4, 15],
        [-10, -5, 10]
    ]

    fig = plt.figure("R = {}, T = {}".format(r, time))

    for i, initial_point in enumerate(initials):
        x, y, z = lorenz.modelling(S, B, r, np.array(initial_point), time)

        ax = fig.add_subplot(220 + i + 1, projection='3d')
        title = ", ".join(map(str, initial_point))

        three_dim_plot(
            ax, title,
            {'x': x, 'y': y, 'z': z},
            {'x': "X", 'y': "Y", 'z': "Z"}
        )


if __name__ == '__main__':
    r = R_CRITICAL + 2
    time = DEFAULT_TIME

    attractor_for_various_r(time)
    plot_attractor_with_projections(r, time)
    plot_volumes(r, time)
    different_initials(r, time)

    plt.show()
