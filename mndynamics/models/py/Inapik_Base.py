import numpy as np
from numpy import exp
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import odeint
from mndynamics.utility import is_sequence


class Inapik(object):
    """
    Inapik neuron model 

    >>> par = {'}
    >>> model = Inapik(par)
    >>> sol = model.simulate()
    >>> plt.plot(sol['t'], sol['v'])

    """

    def __init__(self, par: dict = {}):
        self.check_parameters(par)
        self.set_parameters(par)

    def __call__(self, ):
        print("Inapik neuron model")
        return self._par

    def set_parameters(self, par: dict = {}):

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
            'cm': 1.0,
            'g_k': 10.0,
            'g_na': 20.0,
            'g_l': 8.0,
            'v_k': -90.0,
            'v_na': 60.0,
            'v_l': -80.0,
            'tau_n': 0.15,
            'tau_n_slow': 20.0,
            'g_k_slow': 5.0,
            'v_thr': -20.0,
            'i_ext': 7.0,
            't_end': 100.0,
            'v0': -70.0,
            'dt': 0.01
        }
        return params

    def n_slow_inf(self, v): return 1./(1+exp((-20-v)/5))

    def n_inf(self, v): return 1/(1+exp((-25-v)/5))

    def m_inf(self, v): return 1/(1+exp((-20-v)/15))

    def f_sys(self, x0, t):
        '''
        define Model
        '''
        v, n, n_slow = x0

        dn = (self.n_inf(v)-n)/self.tau_n
        dn_slow = (self.n_slow_inf(v) - n_slow) / self.tau_n_slow
        dv = (self.g_na * self.m_inf(v) * (self.v_na - v) +
              self.g_k * n * (self.v_k - v) +
              self.g_k_slow * n_slow * (self.v_k-v) +
              self.g_l*(self.v_l-v) + self.i_ext)/self.cm

        return np.array([dv, dn, dn_slow])

    def set_initial_state(self):

        x0 = [self.v0,
              self.n_inf(self.v0),
              self.n_slow_inf(self.v0)]
        return x0

    def simulate(self, tspan=None, x0=None):

        x0 = self.set_initial_state() if x0 is None else x0
        tspan = self.tspan if tspan is None else tspan
        sol = odeint(self.f_sys, x0, tspan)

        return {"t": tspan,
                "v":    sol[:, 0],
                "n":    sol[:, 1],
                "n_slow": sol[:, 2]
                }
