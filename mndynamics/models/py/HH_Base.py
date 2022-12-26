import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class HH(object):
    """
    Hudgkin Huxley Model

    Usage:
    >>> par = {'i_ext': 1.5, 't_end': 100.0, 'v0': -70.0, 'dt': 0.01}
    >>> model = HH(par)
    >>> sol = model.simulate()
    >>> plt.plot(sol['t'], sol['v'])

    """

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
            'v_thr':-20.0,
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
        
        # if np.abs(v+45.0) > 1.0e-8:
        return  (v + 45.0) / 10.0 / (1.0 - exp(-(v + 45.0) / 10.0))
        # else:
        #     return 1.0

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

    def simulate(self, tspan=None, x0=None):
        """
        simulate the model

        Parameters
        ----------
        tspan : array
            time span for simulation
        
        Returns
        -------
        dict: {t, v, m, h, n}
            time series of v, m, h, n

        """
        
        x0 = self.set_initial_state() if x0 is None else x0

        tspan = self.tspan if tspan is None else tspan
        sol = odeint(self.f_sys, x0, tspan)

        return {"t": tspan,
                "v":    sol[:, 0],
                "m":    sol[:, 1],
                "h":    sol[:, 2],
                "n":    sol[:, 3]
                }


class HH_Reduced(HH):
    def __init__(self, par={}):
        super().__init__(par)
    
    def __str__(self) -> str:
        return "Reduced Hudgkin Huxley Model"

    def __call__(self) -> None:
        print("Reduced Hudgkin Huxley Model")
        return self._par
    
    def set_initial_state(self):

        x0 = [self.v0,
              self.n_inf(self.v0)]
        return x0

    
    def f_sys(self, x0, t):

        v, n = x0
    
        m = self.m_inf(v)
        h = 0.83 - n

        I_na = -self.g_na * h * m ** 3 * (v - self.v_na)
        I_k = -self.g_k * n ** 4 * (v - self.v_k)
        I_l = -self.g_l * (v - self.v_l)
        
        dv = (self.i_ext + I_na + I_k + I_l) / self.c
        dn = self.alpha_n(v) * (1.0 - n) - self.beta_n(v) * n

        return [dv, dn]

    
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
                "m":    self.m_inf(sol[:, 0]),
                "h":    0.83 - sol[:, 1],
                "n":    sol[:, 1]
                }

class HH_F_I_CURVE(HH):
    def __init__(self, par={}):
        super().__init__(par)
    
    def __str__(self) -> str:
        return "F-I Curve Hudgkin Huxley Model"

    def __call__(self) -> None:
        print("F-I Curve Hudgkin Huxley Model")
        return self._par
    
    def simulate_F_I(self, vec_i_ext, tspan=None):
        '''
        simulate the model with given vector of i_ext and calculate F-I curve
        '''

        tspan = self.tspan if tspan is None else tspan
        num_steps = len(tspan)
        dt = tspan[1] - tspan[0]
        N = int(600 / dt)
        v_thr = self.v_thr
        data = {"i_ext": vec_i_ext}

        for direction in ['forward', 'backward']:
            
            freq = np.zeros(len(vec_i_ext))
            x0 = None
            
            if direction == "backward":
                vec_i_ext = vec_i_ext[::-1]

            for ii in range(len(vec_i_ext)):
                num_spikes = 0
                t_spikes = []
                i_ext = self.i_ext = vec_i_ext[ii]
                
                # set the last state as the initial state

                if ii > 0:
                    x0 = [v[-1], m[-1], h[-1], n[-1]]

                sol = self.simulate(tspan, x0=x0)
                v = sol['v']
                m = sol['m']
                h = sol['h']
                n = sol['n']
                # find steady state
                for i in range(num_steps):
                    if ((i % N) == 0) and (i > 0):
                        maxv = max(v[i - N:i])
                        minv = min(v[i - N:i])
                        maxm = max(m[i - N:i])
                        minm = min(m[i - N:i])
                        maxn = max(n[i - N:i])
                        minn = min(n[i - N:i])
                        maxh = max(h[i - N:i])
                        minh = min(h[i - N:i])
                        if (((maxv - minv) < 0.0001 * abs(maxv + minv)) &
                            ((maxm - minm) < 0.0001 * abs(maxm + minm)) &
                            ((maxh - minh) < 0.0001 * abs(maxh + minh)) &
                                ((maxn - minn) < 0.0001 * abs(maxn + minn))):
                            freq[ii] = 0.0
                            print ("I =%10.3f, f =%10.2f" % (i_ext, freq[ii]))
                            break
                    
                    # spike detection
                    if (i > 0) & (v[i-1] < v_thr) & (v[i] >= v_thr):
                        num_spikes += 1
                        tmp = ((i - 1) * dt * (v[i - 1] - v_thr) +
                            i * dt * (v_thr - v[i])) / (v[i - 1] - v[i])
                        t_spikes.append(tmp)
        
                    if num_spikes == 4:
                        freq[ii] = 1000.0 / (t_spikes[-1] - t_spikes[-2])
                        print ("I =%10.3f, f =%10.2f, t =%18.6f" % (i_ext, freq[ii], tmp))
                        break
            data[direction] = freq
        data['backward'] = data['backward'][::-1]
        return data
    
    def plot_F_I(self, data, ax=None):
        '''
        plot F-I curve
        '''
        ff = data['forward']
        fb = data['backward']
        I = data['i_ext']

        
        ax = plt.gca() if ax is None else ax
        ax.plot(I, ff, 'ro', fillstyle="none", ms=8, label='forward')
        ax.plot(I[::-1], fb[::-1], 'bo', fillstyle="none", ms=8, label='backward')

        index = np.max(np.where(ff < 1e-8)[0])
        I_c = 0.5 * (I[index] + I[index + 1])
        ax.plot([I[index+1], I[index+1]], [0, ff[index+1]], '--b', lw=1)

        index = np.max(np.where(fb < 1e-8)[0])
        I_star = 0.5 * (I[index] + I[index + 1])
        ax.plot([I[index + 1], I[index + 1]], [0, fb[index + 1]], '--b', lw=1)
        ax.text(I_star - 0.1, -15, r"$I_{\ast}$", fontsize=20, color="b")
        ax.text(I_c - 0.1, -15, r"$I_c$", fontsize=20, color="b")

        ax.set_xlabel(r'$I [\mu A/cm^2]$', labelpad=15)
        ax.set_ylabel('frequency [Hz]')
        ax.legend()
        return ax

        

            
                
            
                
        
                        
                






                


