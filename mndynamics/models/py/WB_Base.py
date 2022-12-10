import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class WB(object):

    def __init__(self, par) -> None:

        if not par is None:
            self.check_parameters(par)
        self.set_parameters(par)

    def __call__(self) -> None:
        print("Wang-Buzsaki Model of an Inhibitory Interneuron in Rat Hippocampus")
        return self._par

    def __str__(self) -> str:
        return "Wang-Buzsaki Model of an Inhibitory Interneuron in Rat Hippocampus"

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
            'g_k': 9.0,
            'g_na': 35.0,
            'g_l': 0.1,
            'v_k': -90.0,
            'v_na': 55.0,
            'v_l': -65.0,
            'i_ext': 0.75,
            't_end': 100.0,
            'v0': -63.0,
            'dt': 0.01
        }
        return params

    def alpha_h(self, v):
        return 0.35 * exp(-(v + 58.0) / 20.0)

    def alpha_m(self, v):
        return 0.1 * (v + 35.0) / (1.0 - exp(-0.1 * (v + 35.0)))

    def alpha_n(self, v):
        return -0.05 * (v + 34.0) / (exp(-0.1 * (v + 34.0)) - 1.0)

    def beta_h(self, v):
        return 5.0 / (exp(-0.1 * (v + 28.0)) + 1.0)

    def beta_m(self, v):
        return 4.0 * exp(-(v + 60.0) / 18.0)

    def beta_n(self, v):
        return 0.625 * exp(-(v + 44.0) / 80.0)

    def h_inf(self, v):
        return self.alpha_h(v) / (self.alpha_h(v) + self.beta_h(v))

    def m_inf(self, v):
        return self.alpha_m(v) / (self.alpha_m(v) + self.beta_m(v))

    def n_inf(self, v):
        return self.alpha_n(v) / (self.alpha_n(v) + self.beta_n(v))

    def f_sys(self, x0, t):

        v, h, n = x0

        I_Na = self.g_na * self.m_inf(v) ** 3 * h * (v - self.v_na)
        I_L = self.g_l * (v - self.v_l)
        I_K = self.g_k * n ** 4 * (v - self.v_k)

        dv = -I_Na - I_K - I_L + self.i_ext
        dh = self.alpha_h(v) * (1-h) - self.beta_h(v) * h
        dn = self.alpha_n(v) * (1-n) - self.beta_n(v) * n

        return [dv, dh, dn]

    def simulate(self, tspan=None):
        
        x0 = self.set_initial_state()
        tspan = self.tspan if tspan is None else tspan
        sol = odeint(self.f_sys, x0, tspan)

        return {"t": tspan,
                "v":    sol[:, 0],
                "h":    sol[:, 1],
                "n":    sol[:, 2]
                }

