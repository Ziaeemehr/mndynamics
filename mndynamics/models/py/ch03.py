import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mndynamics.models.py.HH_Base import HH as _HH


class HH_GATING_VARIABLES(_HH):

    def __init__(self, par={}):
        super().__init__(par)

    def plot(self):
        '''
        Plot gating variables and time constants of the HH model
        
        '''

        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(7, 7))
        v = np.arange(-100, 50, 0.01)

        h_inf = [self.h_inf(vi) for vi in v]
        m_inf = [self.m_inf(vi) for vi in v]
        n_inf = [self.n_inf(vi) for vi in v]

        ax[0][0].plot(v, m_inf, lw=2, c="k")
        ax[1][0].plot(v, h_inf, lw=2, c="k")
        ax[2][0].plot(v, n_inf, lw=2, c="k")

        alpha_m = np.array([self.alpha_m(vi) for vi in v])
        beta_m = np.array([self.beta_m(vi) for vi in v])
        alpha_h = np.array([self.alpha_h(vi) for vi in v])
        beta_h = np.array([self.beta_h(vi) for vi in v])
        alpha_n = np.array([self.alpha_n(vi) for vi in v])
        beta_n = np.array([self.beta_n(vi) for vi in v])

        ax[0][1].plot(v, 1.0/(alpha_m+beta_m), lw=2, c="k")
        ax[1][1].plot(v, 1.0/(alpha_h+beta_h), lw=2, c="k")
        ax[2][1].plot(v, 1.0/(alpha_n+beta_n), lw=2, c="k")

        ax[0][0].set_ylabel(r"$m_{\infty} (v)$")
        ax[1][0].set_ylabel(r"$h_{\infty} (v)$")
        ax[2][0].set_ylabel(r"$n_{\infty} (v)$")

        ax[0][1].set_ylabel(r"$\tau_m [ms]$")
        ax[1][1].set_ylabel(r"$\tau_h [ms]$")
        ax[2][1].set_ylabel(r"$\tau_n [ms]$")

        ax[2][0].set_xlabel("v [mV]", fontsize=14)
        ax[2][1].set_xlabel("v [mV]", fontsize=14)

        for i in range(3):
            for j in range(2):
                ax[i][j].set_xlim(min(v), max(v))
                ax[i][j].set_ylim([0, 1.05])
        ax[1][1].set_ylim([0, 10])
        ax[2][1].set_ylim([0, 10])

        plt.tight_layout()

        return fig, ax
