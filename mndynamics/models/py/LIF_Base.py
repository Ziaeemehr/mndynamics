import numpy as np
from numpy import exp
from copy import copy
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class LIF(object):
    """
    Linear Integrate-and-Fire (LIF) Model

    >>> par = {'i_ext': 0.11, 't_end': 100.0, 'v0': -70.0, 'dt': 0.01}
    >>> model = LIF(par)
    >>> sol = model.simulate()
    >>> plt.plot(sol['t'], sol['v'])

    """

    def __init__(self, par={}):

        self.check_parameters(par)
        self.set_parameters(par)

    def __call__(self) -> None:
        print("Linear Integrate-and-Fire (LIF) Model")
        return self._par

    def __str__(self) -> str:
        return "Linear Integrate-and-Fire (LIF) Model"

    def set_parameters(self, par={}):

        self._par = self.get_default_parameters()
        self._par.update(par)

        for key in self._par.keys():
            setattr(self, key, self._par[key])
        self.tspan = np.arange(0.0, self.t_end, self.dt)

    def check_parameters(self, par):
        '''
        Check if the parameters are valid
        '''
        for key in par.keys():
            if key not in self.get_default_parameters().keys():
                raise ValueError('Parameter {} is not valid'.format(key))

    def get_default_parameters(self):

        params = {
            'c': 1.0,
            "tau_m": 10.0,
            't_end': 100.0,
            "i_ext": 0.11,
            'v0': 0.0,
            'dt': 0.01,
            "t_end": 100.0,
        }
        return params

    def set_initial_state(self):

        return self.v0

    def f_sys(self, v):
        return -v / self.tau_m + self.i_ext

    def integrate_rk4(self, x, dt, f):
        k1 = dt * f(x)
        k2 = dt * f(x + 0.5 * k1)
        k3 = dt * f(x + 0.5 * k2)
        k4 = dt * f(x + k3)

        x = x + (k1 + 2.0 * (k2 + k3) + k4) / 6.0
        return x

    def simulate(self, tspan=None):

        x0 = self.set_initial_state()
        tspan = self.tspan if tspan is None else tspan

        num_steps = len(tspan)
        v = np.zeros(num_steps)
        v[0] = x0

        for i in range(1, num_steps):
            v_new = self.integrate_rk4(v[i - 1], self.dt, self.f_sys)

            if v_new <= 1:
                v[i] = v_new
            else:
                v[i] = 0.0

        return {"t": tspan, "v": v}


class LIF_Addapt(LIF):
    """
    Linear Integrate-and-Fire (LIF) Model with Adaptation

    >>> par = {'i_ext': 0.13, 't_end': 100.0, 'v0': -70.0, 'dt': 0.01}
    >>> model = LIF_Addapt(par)
    >>> sol = model.simulate()
    >>> plt.plot(sol['t'], sol['v'])

    """

    def __init__(self, par={}):
        super().__init__(par)

    def __call__(self) -> None:
        print("Linear Integrate-and-Fire (LIF) Model with Adaptation")
        return self._par

    def __str__(self) -> str:
        return "Linear Integrate-and-Fire (LIF) Model with Adaptation"

    def get_default_parameters(self):

        params = {
            'c': 1.0,
            "tau_m": 10.0,
            "tau_w": 40.0,
            "delta": 0.05,
            "i_ext": 0.13,
            'v0': 0.0,
            'dt': 0.01,
            "t_end": 300.0,
        }
        return params

    def set_initial_state(self):

        return [self.v0, 0.0]

    def f_sys(self, x0):

        v, w = x0
        dv = -v / self.tau_m + self.i_ext - w*v
        dw = -w/self.tau_w

        return np.array([dv, dw])

    def simulate(self, tspan=None):

        x0 = self.set_initial_state()
        tspan = self.tspan if tspan is None else tspan

        num_steps = len(tspan)
        y = np.zeros((num_steps, 2))

        for i in range(1, num_steps):
            y_new = self.integrate_rk4(y[i - 1, :], self.dt, self.f_sys)
            v_new = y_new[0]
            w_new = y_new[1]

            if v_new <= 1:
                y[i, :] = copy(y_new)
            else:
                y[i, 0] = 0.0
                y[i, 1] = y[i-1, 1] + self.delta

                # t_old = tspan[i-1]
                # t_new = tspan[i]
                # t_spike = (v_new - 1) * t_old + (1 - y[i-1, 0]) * t_new
                # t_spike = t_spike / (v_new - y[i-1, 0])
                # y[i, 0] = 0.0
                # y[i, 1] = ((v_new - 1) * y[i-1, 1] + (1 - y[i-1, 0])
                #            * w_new) / (v_new - y[i-1, 0])


        return {"t": tspan, "v": y[:, 0], "w": y[:, 1]}
