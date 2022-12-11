import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class HH(object):

    def __init__(self, par={}):

        
        self.check_parameters(par)
        self.set_parameters(par)

    def __call__(self) -> None:
        print("Hudgkin Huxley Model")
        return self._par

    def __str__(self) -> str:
        return "Hudgkin Huxley Model"

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
            'g_k': 36.0,
            'g_na': 120.0,
            'g_l': 0.3,
            'v_k': -82.0,
            'v_na': 45.0,
            'v_l': -59.0,
            'i_ext': 7.0,
            't_end': 50.0,
            'v0': -70.0,
            'dt': 0.01
        }
        return params

    def set_initial_state(self):

        x0 = [self.v0,
              self.m_inf(self.v0),
              self.h_inf(self.v0),
              self.n_inf(self.v0)]
        return x0

    def beta_n(self, v):
        return 0.125 * exp(-(v + 70.0) / 80.0)

    def beta_m(self, v):
        return 4.0 * exp(-(v + 70.0) / 18.0)

    def beta_h(self, v):
        return 1. / (exp(-(v + 40.0) / 10.0) + 1.0)

    def alpha_n(self, v):
        return 0.01 * (-60.0 - v) / (exp((-60.0 - v) / 10.0) - 1.0)

    def alpha_m(self, v):
        
        if np.abs(v+45.0) > 1.0e-8:
            return  (v + 45.0) / 10.0 / (1.0 - exp(-(v + 45.0) / 10.0))
        else:
            return 1.0

    def alpha_h(self, v):
        return 0.07*exp(-(v+70)/20)

    def h_inf(self, v):
        return self.alpha_h(v) / (self.alpha_h(v) + self.beta_h(v))

    def m_inf(self, v):
        return self.alpha_m(v) / (self.alpha_m(v) + self.beta_m(v))

    def n_inf(self, v):
        return self.alpha_n(v) / (self.alpha_n(v) + self.beta_n(v))

    def f_sys(self, x0, t):
        '''
        define HH Model
        '''
        v, m, h, n = x0
        dv = (self.i_ext - self.g_na * m**3 * h * (v - self.v_na) -
              self.g_k * n**4 * (v - self.v_k) - self.g_l * (v - self.v_l)) / self.c
        dm = self.alpha_m(v) * (1.0 - m) - self.beta_m(v) * m
        dh = self.alpha_h(v) * (1.0 - h) - self.beta_h(v) * h
        dn = self.alpha_n(v) * (1.0 - n) - self.beta_n(v) * n
        return [dv, dm, dh, dn]

    def simulate(self, tspan=None):
        
        x0 = self.set_initial_state()

        tspan = self.tspan if tspan is None else tspan
        sol = odeint(self.f_sys, x0, tspan)

        return {"t": tspan,
                "v":    sol[:, 0],
                "m":    sol[:, 1],
                "h":    sol[:, 2],
                "n":    sol[:, 3]
                }
