import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mndynamics.models.py.HH_Base import HH as _HH


class HH_SOLUTION(_HH):

    def __init__(self, par={}):
        super().__init__(par)


class HH_REFRACTORINESS(_HH):

    def __init__(self, par=None):
        super().__init__(par)
    
    def f_sys(self, x0, t, i_ext, PULSE_ONSET):
        '''
        define HH Model
        '''
        v, m, h, n = x0

        if (t > PULSE_ONSET) & (t < PULSE_ONSET + 1):
            i_ext = 40.0

        dv = (i_ext - self.g_na * m**3 * h * (v - self.v_na) -
              self.g_k * n**4 * (v - self.v_k) - self.g_l * (v - self.v_l)) / self.c
        dm = self.alpha_m(v) * (1.0 - m) - self.beta_m(v) * m
        dh = self.alpha_h(v) * (1.0 - h) - self.beta_h(v) * h
        dn = self.alpha_n(v) * (1.0 - n) - self.beta_n(v) * n
        return [dv, dm, dh, dn]
    
    def simulate(self, tspan=None, *args):
        
        x0 = self.set_initial_state()

        tspan = self.tspan if tspan is None else tspan
        sol = odeint(self.f_sys, x0, tspan, args=args)

        return {"t": tspan,
                "v":    sol[:, 0],
                "m":    sol[:, 1],
                "h":    sol[:, 2],
                "n":    sol[:, 3]
                }
