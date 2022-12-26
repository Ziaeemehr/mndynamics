import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class FN(object):
    def __init__(self, par={}) -> None:
        self.check_parameters(par)
        self.set_parameters(par)

    def __str__(self) -> str:
        return "The FitzHugh-Nagumo Model."

    def __call__(self) -> None:
        print("The FitzHugh-Nagumo Model.")
        return self._par

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
            'a':1.25,
            'tau_n':15.625,
            "i_ext":-0.5,
            't_end': 400.0,
            'dt': 0.01
        }
        return params

    def set_initial_state(self, x0=None):

        x0 = [-1,-2] if x0 is None else x0
        return np.array(x0)
    
    def f_sys(self, x0, t):
        '''
        define FN Model
        '''
        v, n = x0
        dv = v - (v**3)/3 - n + self.i_ext
        dn = (self.a*v - self.a*n)/self.tau_n
        return np.array([dv, dn])

    def simulate(self, tspan=None, x0=None):
        """
        simulate the model

        Parameters
        ----------
        tspan : array
            time span for simulation
        
        Returns
        -------
        dict: {t, v, n}
            time series of v, n

        """
        
        x0 = self.set_initial_state() if x0 is None else x0
        tspan = self.tspan if tspan is None else tspan
        sol = odeint(self.f_sys, x0, tspan)

        return {"t": tspan,
                "v":    sol[:, 0],
                "n":    sol[:, 1]
                }


        