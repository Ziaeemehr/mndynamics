import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from mndynamics.models.py.WB_Base import WB as _WB
from mndynamics.models.py.RTM_Base import RTM as _RTM
from mndynamics.models.py.Erisir_Base import Erisir as _Erisir

class PLOT(object):

    def plot(self, ax=None, **kwargs):

        if ax is None:
            fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(7, 7))
        
        v = np.arange(-100, 50, 0.01)


        ax[0][0].plot(v, self.m_inf(v), **kwargs)
        ax[1][0].plot(v, self.h_inf(v), **kwargs)
        ax[2][0].plot(v, self.n_inf(v), **kwargs)

        ax[0][1].plot(v, 1.0 / (self.alpha_m(v) + self.beta_m(v)), **kwargs)

        ax[1][1].plot(v, 1.0/(self.alpha_h(v)+self.beta_h(v)), **kwargs)
        ax[2][1].plot(v, 1.0/(self.alpha_n(v)+self.beta_n(v)), **kwargs)

        ax[0][0].set_ylabel(r"$m_{\infty} (v)$")
        ax[1][0].set_ylabel(r"$h_{\infty} (v)$")
        ax[2][0].set_ylabel(r"$n_{\infty} (v)$")

        ax[0][1].set_ylabel(r"$\tau_m [ms]$")
        ax[1][1].set_ylabel(r"$\tau_h [ms]$")
        ax[2][1].set_ylabel(r"$\tau_n [ms]$")

        ax[2][0].set_xlabel("v [mV]", fontsize=14)
        ax[2][1].set_xlabel("v [mV]", fontsize=14)
        ax[0][0].legend(frameon=False, fontsize=12)

        for i in range(3):
            for j in range(2):
                ax[i][j].set_xlim(min(v), max(v))
                ax[i][j].set_ylim([0, 1.05])
        ax[1][1].set_ylim([0, 20])
        ax[2][1].set_ylim([0, 20])
        plt.tight_layout()


class RTM_GATING_VARIABLES(_RTM, PLOT):    
    def __init__(self, par=None):
        super().__init__(par)
    

class WB_GATING_VARIABLES(_WB, PLOT):
    def __init__(self, par=None):
        super().__init__(par)
    
class Erisir_GATING_VARIABLES(_Erisir, PLOT):
    def __init__(self, par=None):
        super().__init__(par)

    