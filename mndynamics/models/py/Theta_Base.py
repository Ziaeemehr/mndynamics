import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class Theta(object):

    def __init__(self, par={}):

        self.check_parameters(par)
        self.set_parameters(par)

    def __call__(self) -> None:
        print("Theta Neuron Model")
        return self._par

    def __str__(self) -> str:
        return "Theta Neuron Model"

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
            "tau_m": 0.5,
            't_end': 100.0,
            "i_ext": 0.505,
            'v0': 0.0,
            'dt': 0.01,
            't_end':150.0,
        }
        return params

    def set_initial_state(self):

        return self.v0

    def f_sys(self, theta, t):
        return -np.cos(theta)/self.tau_m+2*self.i_ext*(1.0+np.cos(theta))

    def simulate(self, tspan=None):
        
        x0 = self.set_initial_state()
        tspan = self.tspan if tspan is None else tspan
        sol = odeint(self.f_sys, x0, tspan)

        return {"t": tspan, "v": sol}

