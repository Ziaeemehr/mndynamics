import numpy as np
from numpy import exp
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mndynamics.utility import is_sequence


class WC(object):

    def __init__(self, par: dict = {}) -> None:

        self.check_parameters(par)
        self.set_parameters(par)

    def __call__(self, ):
        print("Wilson-Cowan Model")
        return self._par

    def __str__(self) -> str:
        return "Wilson-Cowan Model"

    def set_parameters(self, par={}):

        self._par = self.get_default_parameters()
        self._par.update(par)

        for key in self._par.keys():
            setattr(self, key, self._par[key])
        self.tspan = np.arange(0.0, self.t_end, self.dt)

    def set_initial_state(self, x0=None):
        if x0 is None:
            return [self.E0, self.I0]
        else:
            return x0

    def check_parameters(self, par):
        '''
        Check if the parameters are valid
        '''
        for key in par.keys():
            if key not in self.get_default_parameters().keys():
                raise ValueError('Parameter {} is not valid'.format(key))

    def get_default_parameters(self):

        params = {
            "I_E": 20.0,
            "I_I": 0.0,
            "w_EE": 1.5,
            "w_IE": 1.0,
            "w_EI": 1.0,
            "w_II": 0.0,
            "tau_E": 5.0,
            "tau_I": 10.0,
            "t_end": 300.0,
            "E0": 50.0,
            "I0": 10.0,
            "dt": 0.01,
        }

        return params

    def g(self, x):
        return 100 * x * x / (400.0 + x * x) * (x > 0)

    def f(self, x):
        return 100.0 * x * x / (900 + x * x) * (x > 0)

    def dE(self, E, I):
        return (self.f(self.w_EE * E - self.w_IE * I + self.I_E) - E) / self.tau_E

    def dI(self, E, I):
        return (self.g(self.w_EI * E - self.w_II * I + self.I_I) - I) / self.tau_I

    def f_sys(self, x, t):
        E, I = x

        return [self.dE(E, I), self.dI(E, I)]

    def simulate(self, tspan=None, x0=None):

        x0 = self.set_initial_state() if x0 is None else x0
        tspan = self.tspan if tspan is None else tspan
        sol = odeint(self.f_sys, x0, tspan)

        return {"t": tspan,
                "E":    sol[:, 0],
                "I":    sol[:, 1],
                }

    def inv_f(self, x):
        return np.sqrt(900 * x / (100.0 - x)) * (x < 100.0)

    def inv_g(self, x):
        return np.sqrt(400 * x / (100.0 - x)) * (x < 100.0)

    def plot_streamplot(self,
                        ax,
                        f_sys,
                        x=[0, 100],
                        y=[0, 100],
                        dx=0.1,
                        dy=0.1):

        phi1, phi2 = np.meshgrid(np.arange(x[0], x[1], dx),
                                 np.arange(y[0], y[1], dy))
        dphi1_dt, dphi2_dt = f_sys([phi1, phi2], 0)
        ax.streamplot(phi1, phi2,
                      dphi1_dt, dphi2_dt,
                      color='k',
                      linewidth=0.5,
                      cmap=plt.cm.autumn)

    def plot_nullcline(self, f, ax, label=None, color="r"):
        x = 100.0
        I_left = np.arange(0, x, 0.1)
        I_right = np.arange(0.1, x + 0.1, 0.1)
        nx = len(I_left)
        i_red_index = 0
        E_red = []
        I_red = []
        # bisection method to find zero
        for i in range(nx):
            E = i / float(nx) * 100.0
            R_left = f(E, I_left)
            R_right = f(E, I_right)
            ind = np.where((R_left * R_right) < 0)[0]
            if len(ind) > 0:
                for j in range(len(ind)):
                    I_l = I_left[ind[j]]
                    I_r = I_right[ind[j]]
                    while (I_r-I_l) > 1e-8:
                        I_c = 0.5 * (I_r + I_l)
                        R_l = f(E, I_l)
                        R_c = f(E, I_c)

                        if (R_l * R_c) < 0:
                            I_r = I_c
                        else:
                            I_l = I_c

                        I_c = 0.5 * (I_r + I_l)
                        i_red_index = i_red_index + 1.0
                        E_red.append(E)
                        I_red.append(I_c)

        ax.plot(E_red, I_red, lw=2, c=color, ls="--", label=label)
        if label:
            ax.legend()

    def plot_phase_plane(self, E_lim=[0, 100], I_lim=[0, 100]):

        fig, ax = plt.subplots(figsize=(5, 5))
        self.plot_streamplot(ax, self.f_sys, E_lim, I_lim, 0.1, 0.1)
        self.plot_nullcline(self.dE, ax, label="dE/dt=0", color="r")
        self.plot_nullcline(self.dI, ax, label="dI/dt=0", color="b")
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])

        ax.set_xlabel("E", fontsize=14)
        ax.set_ylabel("I", fontsize=14)
        ax.tick_params(labelsize=14)
        plt.tight_layout()
