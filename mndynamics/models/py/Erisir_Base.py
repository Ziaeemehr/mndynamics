import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class Erisir(object):

    def __init__(self, par) -> None:

        if not par is None:
            self.check_parameters(par)
        self.set_parameters(par)

    def __call__(self) -> None:
        print("")
        return self._par

    def __str__(self) -> str:
        return ""

    def set_parameters(self, par=None):

        self._par = self.get_default_parameters()
        if not par is None:
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
            'g_k': 224.0,
            'g_na': 112.0,
            'g_l': 0.1,
            'v_k': -90.0,
            'v_na': 60.0,
            'v_l': -70.0,
            'i_ext': 7.0,
            't_end': 100.0,
            'v0': -70.0,
            'dt': 0.01
        }
        return params

    def alpha_h(self, v):
        return 0.0035 / exp(v / 24.186)

    def alpha_m(self, v):
        return 40 * (75.5 - v) / (exp((75.5 - v) / 13.5) - 1)

    def alpha_n(self, v):
        return (95 - v) / (exp((95 - v) / 11.8) - 1)

    def beta_h(self, v):
        return - 0.017 * (v + 51.25) / (exp(-(v + 51.25) / 5.2) - 1)

    def beta_m(self, v):
        return 1.2262 / exp(v / 42.248)

    def beta_n(self, v):
        return 0.025 / exp(v / 22.222)

    def h_inf(self, v):
        return self.alpha_h(v) / (self.alpha_h(v) + self.beta_h(v))

    def m_inf(self, v):
        return self.alpha_m(v) / (self.alpha_m(v) + self.beta_m(v))

    def n_inf(self, v):
        return self.alpha_n(v) / (self.alpha_n(v) + self.beta_n(v))

    def f_sys(self, x0, t):
        '''
        define Erisir Model
        '''
        v, n, h, = x0
        m = self.alpha_m(v) / (self.alpha_m(v) + self.beta_m(v))
        dv = self.i_ext - self.g_na * h * m ** 3 * \
            (v - self.v_na) - self.g_k * n ** 2 * \
            (v - self.v_k) - self.g_l * (v - self.v_l)
        dn = self.alpha_n(v) * (1.0 - n) - self.beta_n(v) * n
        dh = self.alpha_h(v) * (1.0 - h) - self.beta_h(v) * h

        return [dv, dn, dh]
