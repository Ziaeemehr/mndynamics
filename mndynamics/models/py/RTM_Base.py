import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class RTM(object):
    """
    Reduced Traub-Miles Model of a Pyramidal Neuron in Rat Hippocampus
    
    Usage:
    >>> par = {'i_ext': 1.5, 't_end': 100.0, 'v0': -70.0, 'dt': 0.01}
    >>> model = RTM(par)
    >>> sol = model.simulate()
    >>> plt.plot(sol['t'], sol['v'])

    """

    def __init__(self, par: dict = {}) -> None:

        self.check_parameters(par)
        self.set_parameters(par)

    def __call__(self) -> None:
        print("Reduced Traub-Miles Model of a Pyramidal Neuron in Rat Hippocampus")
        return self._par

    def __str__(self) -> str:
        return "Reduced Traub-Miles Model of a Pyramidal Neuron in Rat Hippocampus"

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
            'c': 1.0,
            'g_k': 80.0,
            'g_na': 100.0,
            'g_l': 0.1,
            'v_k': -100.0,
            'v_na': 50.0,
            'v_l': -67.0,
            'i_ext': 1.5,
            't_end': 100.0,
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

    def alpha_h(self, v):
        return 0.128 * exp(-(v + 50.0) / 18.0)

    def alpha_m(self, v):
        return 0.32 * (v + 54) / (1.0 - exp(-(v + 54.0) / 4.0))

    def alpha_n(self, v):
        return 0.032 * (v + 52) / (1.0 - exp(-(v + 52.0) / 5.0))

    def beta_h(self, v):
        return 4.0 / (1.0 + exp(-(v + 27.0) / 5.0))

    def beta_m(self, v):
        return 0.28 * (v + 27.0) / (exp((v + 27.0) / 5.0) - 1.0)

    def beta_n(self, v):
        return 0.5 * exp(-(v + 57.0) / 40.0)

    def h_inf(self, v):
        return self.alpha_h(v) / (self.alpha_h(v) + self.beta_h(v))

    def m_inf(self, v):
        return self.alpha_m(v) / (self.alpha_m(v) + self.beta_m(v))

    def n_inf(self, v):
        return self.alpha_n(v) / (self.alpha_n(v) + self.beta_n(v))

    def f_sys(self, x0, t):
        '''
        define RTM Model
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


class RTM_M(RTM):
    """
    Reduced Traub-Miles neuron model with M-current
    Simulate the model with parameters given in the dictionary par:

    Usage:
    >>> par = {'i_ext': 1.5, 't_end': 100.0, 'v0': -70.0, 'dt': 0.01}
    >>> model = RTM_M(par)
    >>> sol = model.simulate()
    >>> plt.plot(sol['t'], sol['v'])


    """
    def __init__(self, par: dict = {}) -> None:
        super().__init__(par)

    def __str__(self) -> str:
        return "Reduced Traub-Miles Model with M-current"

    def __call__(self) -> None:
        print("Reduced Traub-Miles Model with M-current")
        return self._par

    def set_initial_state(self):

        x0 = [self.v0,
              #   self.m_inf(self.v0),
              self.h_inf(self.v0),
              self.n_inf(self.v0),
              0.0]
        return x0

    def get_default_parameters(self):

        params = {
            'c': 1.0,
            'g_k': 80.0,
            'g_na': 100.0,
            'g_l': 0.1,
            'g_m': 0.25,
            'v_k': -100.0,
            'v_na': 50.0,
            'v_l': -67.0,
            'i_ext': 1.5,
            't_end': 100.0,
            'v0': -70.0,
            'dt': 0.01,
        }
        return params

    def w_inf(self, v):
        return 1.0/(1.0 + exp(-(v + 35.0)/10.0))

    def tau_w(self, v):
        return 400.0/(3.3*exp((v+35)/20)+exp(-(v+35)/20))

    def f_sys(self, x0, t):
        '''
        define RTM Model with M-current
        '''
        v, h, n, w = x0
        dv = (self.i_ext -
              self.g_na * self.m_inf(v)**3 * h * (v - self.v_na) -
              self.g_k * n**4 * (v - self.v_k) -
              self.g_m * w * (v - self.v_k) -
              self.g_l * (v - self.v_l)) / self.c
        dh = self.alpha_h(v) * (1.0 - h) - self.beta_h(v) * h
        dn = self.alpha_n(v) * (1.0 - n) - self.beta_n(v) * n
        dw = (self.w_inf(v) - w) / self.tau_w(v)
        return [dv, dh, dn, dw]

    def simulate(self, tspan=None):

        """
        simulate the model

        Parameters
        ----------
        tspan : array_like
            time span

        Returns
        -------
        dict : 
            time series of membrane potential (v), activation variables 
            m, h, n and w
        """

        x0 = self.set_initial_state()
        tspan = self.tspan if tspan is None else tspan
        sol = odeint(self.f_sys, x0, tspan)

        return {"t": tspan,
                "v":    sol[:, 0],
                "m":    self.m_inf(sol[:, 0]),
                "h":    sol[:, 1],
                "n":    sol[:, 2],
                "w":    sol[:, 3]
                }


class RTM_AHP(RTM):
    def __init__(self, par: dict = {}) -> None:
        super().__init__(par)

    def __str__(self) -> str:
        return "Reduced Traub-Miles Model with AHP-current"

    def __call__(self) -> None:
        print("Reduced Traub-Miles Model with AHP-current")
        return self._par

    def set_initial_state(self):

        x0 = [self.v0,
              self.h_inf(self.v0),
              self.n_inf(self.v0),
              0.0]
        return x0

    def get_default_parameters(self):

        params = {
            'c': 1.0,
            'g_k': 80.0,
            'g_na': 100.0,
            'g_l': 0.1,
            'g_ahp': 0.25,
            'v_k': -100.0,
            'v_na': 50.0,
            'v_l': -67.0,
            'i_ext': 1.5,
            't_end': 100.0,
            'v0': -70.0,
            'dt': 0.01,
        }
        return params

    def ca_inf(self, v):
        return (120-v)/(1+exp(-(v+15)/5))*4/25

    def tau_w(self, v):
        return 400.0/(3.3*exp((v+35)/20)+exp(-(v+35)/20))

    def f_sys(self, x0, t):
        '''
        define RTM Model with AHP-current
        '''
        v, h, n, ca = x0
        dv = (self.i_ext -
              self.g_na * self.m_inf(v)**3 * h * (v - self.v_na) -
              self.g_k * n**4 * (v - self.v_k) -
              self.g_ahp * ca * (v - self.v_k) -
              self.g_l * (v - self.v_l)) / self.c
        dh = self.alpha_h(v) * (1.0 - h) - self.beta_h(v) * h
        dn = self.alpha_n(v) * (1.0 - n) - self.beta_n(v) * n
        dca = (self.ca_inf(v) - ca) / 80.0

        return [dv, dh, dn, dca]

    def simulate(self, tspan=None):

        x0 = self.set_initial_state()
        tspan = self.tspan if tspan is None else tspan
        sol = odeint(self.f_sys, x0, tspan)

        return {"t": tspan,
                "v":    sol[:, 0],
                "m":    self.m_inf(sol[:, 0]),
                "h":    sol[:, 1],
                "n":    sol[:, 2],
                "ca":    sol[:, 3]
                }
