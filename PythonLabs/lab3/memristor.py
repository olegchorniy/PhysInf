import numpy as np

from labs.plotting import plot_on


# noinspection PyUnresolvedReferences
class Memristor(object):
    _title_counter = 0

    def __init__(self, D, mu, nu, w0, R_ON, R_OFF, U, Phi):
        self._D = D
        self._mu = mu
        self._nu = nu
        self._w0 = w0
        self._R_ON = R_ON
        self._R_OFF = R_OFF

        self.U = U
        self.Phi = Phi

        self._w_init = (w0 ** 2) * (R_ON - R_OFF) / (2 * D) + w0 * R_OFF
        self._title_counter = 0

    def w(self, t):
        D = self._D
        mu = self._mu
        R_ON = self._R_ON
        R_OFF = self._R_OFF
        w_init = self._w_init

        # Discriminant
        d = R_OFF ** 2 + 2 * ((R_ON - R_OFF) / D) * (w_init + mu * (R_ON / D) * self.Phi(t))
        enumerator = -R_OFF + d ** 0.5
        denominator = (R_ON - R_OFF) / D

        values = enumerator / denominator

        values[values >= D] = D
        values[values <= 0] = 0

        return values

    def I(self, t):
        D = self._D
        R_ON = self._R_ON
        R_OFF = self._R_OFF

        U_of_t = self.U(t)
        w_of_t = self.w(t)

        return D * U_of_t / (R_ON * w_of_t + R_OFF * (D - w_of_t))

    def q(self, t):
        I = self.I(t)
        q = [0]

        q_prev = q[0]
        t_prev = t[0]

        for t_curr, i in zip(t[1:], I[:-1]):
            q.append(q_prev + i * (t_curr - t_prev))

            t_prev = t_curr
            q_prev = q[-1]

        return q[1:]

    def task1(self, plt, N, periods):
        T = np.pi * periods / self._nu
        TIME = np.linspace(0, T, N)

        fig, axes = plt.subplots(3, 1, num=self._get_title(self, 'Task 1'), sharex=True)

        axes[0].plot(TIME, self.w(TIME) / self._D)
        axes[0].set_title(r'$\frac{\omega\left( t \right)}{D}$')

        axes[1].plot(TIME, self.I(TIME))
        axes[1].set_title(r'$I\left( t \right)$')

        axes[2].plot(TIME, self.U(TIME))
        axes[2].set_title(r'$U\left( t \right)$')

        plt.xlabel('$t$')

    def task2(self, plt, N, periods):
        T = np.pi * periods / self._nu
        TIME = np.linspace(0, T, N)

        axes = plt.figure(self._get_title(self, 'Task 2')).gca()
        plot_on(axes, self.U(TIME), self.I(TIME), r'$I\left( U \right)$', '$U$', '$I$')

    def task3(self, plt, N, periods):
        T = np.pi * periods / self._nu
        TIME = np.linspace(0, T, N)

        axes = plt.figure(self._get_title(self, 'Task 3')).gca()
        plot_on(axes, self.Phi(TIME)[1:], self.q(TIME), r'$q\left( \Phi \right)$', '$\Phi$', '$q$')

    @classmethod
    def _get_title(cls, self, prefix):
        cls._title_counter += 1
        title_num = cls._title_counter
        return ('#{}, ' + prefix + ', R_OFF = {}').format(title_num, self._R_OFF)
