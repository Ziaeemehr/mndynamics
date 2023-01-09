import numpy as np
from tqdm import tqdm
from numpy import exp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mndynamics.utility import is_sequence


class Erisir(object):

    def __init__(self, par={}) -> None:

        self.check_parameters(par)
        self.set_parameters(par)

    def __call__(self) -> None:
        print("Erisir Model of an Inhibitory Interneuron in Mouse Cortex")
        return self._par

    def __str__(self) -> str:
        return "Erisir Model of an Inhibitory Interneuron in Mouse Cortex"

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
            'g_k': 224.0,
            'g_na': 112.0,
            'g_l': 0.5,
            'v_k': -90.0,
            'v_na': 60.0,
            'v_l': -70.0,
            'v_thr': -20.0,
            'i_ext': 7.0,
            't_end': 100.0,
            'v0': -70.0,
            'dt': 0.01
        }
        return params

    def set_initial_state(self):

        x0 = [self.v0,
              self.n_inf(self.v0),
              self.h_inf(self.v0)]
        return x0

    def alpha_h(self, v):
        return 0.0035 / exp(v / 24.186)

    def alpha_m(self, v):
        return 40 * (75.5 - v) / (exp((75.5 - v) / 13.5) - 1)

    def alpha_n(self, v):
        return (95 - v) / (exp((95 - v) / 11.8) - 1)

    def beta_h(self, v):
        return - 0.017 * (v + 51.25) / (exp(-(v + 51.25) / 5.2) - 1)

    def beta_m(self, v):
        return 1.2262 / exp(v / 42.248)

    def beta_n(self, v):
        return 0.025 / exp(v / 22.222)

    def h_inf(self, v):
        return self.alpha_h(v) / (self.alpha_h(v) + self.beta_h(v))

    def m_inf(self, v):
        return self.alpha_m(v) / (self.alpha_m(v) + self.beta_m(v))

    def n_inf(self, v):
        return self.alpha_n(v) / (self.alpha_n(v) + self.beta_n(v))

    def f_sys(self, x0, t):
        '''
        define Erisir Model
        '''
        v, n, h, = x0
        m = self.alpha_m(v) / (self.alpha_m(v) + self.beta_m(v))
        dv = self.i_ext - self.g_na * h * m ** 3 * \
            (v - self.v_na) - self.g_k * n ** 2 * \
            (v - self.v_k) - self.g_l * (v - self.v_l)
        dn = self.alpha_n(v) * (1.0 - n) - self.beta_n(v) * n
        dh = self.alpha_h(v) * (1.0 - h) - self.beta_h(v) * h

        return [dv, dn, dh]

    def simulate(self, tspan=None, x0=None):

        x0 = self.set_initial_state() if x0 is None else x0
        tspan = self.tspan if tspan is None else tspan
        sol = odeint(self.f_sys, x0, tspan)

        return {"t": tspan,
                "v":    sol[:, 0],
                "h":    sol[:, 1],
                "n":    sol[:, 2]
                }


class ErisirM(Erisir):
    def __init__(self, par={}) -> None:
        super().__init__(par)

    def __call__(self) -> None:
        print("Modified Erisir Model of an Inhibitory Interneuron in Mouse Cortex")
        return self._par

    def __str__(self) -> str:
        return "Modified Erisir Model of an Inhibitory Interneuron in Mouse Cortex"

    def f_sys(self, x0, t):
        '''
        define Modified Erisir Model
        '''
        v, n, h = x0
        m = self.alpha_m(v) / (self.alpha_m(v) + self.beta_m(v))
        dv = self.i_ext - self.g_na * h * m ** 3 * \
            (v - self.v_na) - self.g_k * n ** 4 * \
            (v - self.v_k) - self.g_l * (v - self.v_l)
        dn = self.alpha_n(v) * (1.0 - n) - self.beta_n(v) * n
        dh = self.alpha_h(v) * (1.0 - h) - self.beta_h(v) * h

        return [dv, dn, dh]


class Erisir_F_I_CURVE(Erisir):

    def __init__(self, par={}) -> None:
        super().__init__(par)

    def __call__(self):
        print("F-I Curve of Erisir Model of an Inhibitory Interneuron in Mouse Cortex")
        return self._par

    def __str__(self) -> str:
        return "F-I Curve of Erisir Model of an Inhibitory Interneuron in Mouse Cortex"

    def simulate_F_I(self, vec_i_ext, tspan=None, directions='both'):
        '''
        Simulate the F-I curve with a sequence of i_ext

        Parameters
        ----------
        vec_i_ext : sequence
            Sequence of i_ext values    
        tspan : sequence, optional
            Time span of the simulation. The default is None.
        direction : str, optional
            Direction of the simulation. The default is None.
            options: 'forward', 'backward', 'both'
        '''

        assert(is_sequence(vec_i_ext)), "vec_i_ext must be a sequence"
        tspan = self.tspan if tspan is None else tspan
        num_steps = len(tspan)
        dt = tspan[1] - tspan[0]
        N = int(1000 / dt)
        v_thr = self.v_thr
        data = {"i_ext": vec_i_ext}
        if directions == 'both':
            directions = ['forward', 'backward']
        else:
            directions = [directions]

        for direction in directions:
            freq = np.zeros(len(vec_i_ext))
            x0 = None

            if direction == "backward":
                vec_i_ext = vec_i_ext[::-1]

            for ii in tqdm(range(len(vec_i_ext)), desc=direction):
                num_spikes = 0
                t_spikes = []
                i_ext = self.i_ext = vec_i_ext[ii]

                # set the last state as the initial state
                if ii > 0:
                    x0 = [v[-1], h[-1], n[-1]]

                sol = self.simulate(tspan, x0=x0)
                v = sol['v']
                h = sol['h']
                n = sol['n']
                # find steady state
                for i in range(num_steps):
                    if ((i % N) == 0) and (i > 0):
                        maxv = max(v[i - N:i])
                        minv = min(v[i - N:i])
                        maxn = max(n[i - N:i])
                        minn = min(n[i - N:i])
                        maxh = max(h[i - N:i])
                        minh = min(h[i - N:i])
                        if (((maxv - minv) < 0.0001 * abs(maxv + minv)) &
                            ((maxh - minh) < 0.0001 * abs(maxh + minh)) &
                                ((maxn - minn) < 0.0001 * abs(maxn + minn))):
                            freq[ii] = 0.0
                            # print ("I =%10.3f, f =%10.2f" % (i_ext, freq[ii]))
                            break

                    # spike detection
                    if (i > 0) & (v[i-1] < v_thr) & (v[i] >= v_thr):
                        num_spikes += 1
                        tmp = ((i - 1) * dt * (v[i - 1] - v_thr) +
                               i * dt * (v_thr - v[i])) / (v[i - 1] - v[i])
                        t_spikes.append(tmp)

                    if num_spikes == 4:
                        freq[ii] = 1000.0 / (t_spikes[-1] - t_spikes[-2])
                        # print ("I =%10.3f, f =%10.2f" % (i_ext, freq[ii]))
                        break
            data[direction] = freq
        if "backward" in data.keys():
            data['backward'] = data['backward'][::-1]
        return data

    def plot_F_I(self, data, ax=None):
        '''
        plot F-I curve
        '''
        directions = list(data.keys())
        directions.remove('i_ext')

        ax = plt.gca() if ax is None else ax
        for direction in directions:
            f = data[direction]
            I = data['i_ext']
            if direction == 'forward':
                ax.plot(I, f, 'ro', fillstyle="none", ms=8, label='forward')
            elif direction == 'backward':
                ax.plot(I[::-1], f[::-1], "bo",
                        fillstyle="none", label='backward')
            else:
                raise ValueError("direction must be 'forward' or 'backward'")

        ax.set_xlabel(r'$I [\mu A/cm^2]$', labelpad=15)
        ax.set_ylabel('frequency [Hz]')
        ax.legend()
        return ax


class Erisir_Burst(Erisir):
    '''
    The Erisir neuron with a slow potassium current that is strengthened by firing
    which turns the neuron into a burster.

    Reference
    ---------
    An introduction to modeling neuronal dynamics, Borgers, Chapter 19.
    '''

    def __init__(self, par={}) -> None:
        super().__init__(par)

    def __str__(self) -> str:
        return "Bursty Erisir Model"

    def __call__(self) -> None:
        print("Bursty Erisir Model")
        return self._par

    def get_default_parameters(self):

        params = super().get_default_parameters()
        params.update({'g_k_slow': 1.5,
                       'i_ext': 7.5,
                       "tau_n_slow": 100.0})
        return params

    def set_initial_state(self):

        x0 = [self.v0,
              self.n_inf(self.v0),
              self.h_inf(self.v0),
              self.n_slow_inf(self.v0)
              ]
        return x0

    def n_slow_inf(self, v):
        return 1.0 / (1.0 + np.exp((-20.0 - v) / 5.0))

    def f_sys(self, x0, t):

        v, n, h, n_slow = x0
        m = self.alpha_m(v) / (self.alpha_m(v) + self.beta_m(v))
        dv = self.i_ext - \
            self.g_na * h * m ** 3 * (v - self.v_na) - \
            self.g_k * n ** 2 * (v - self.v_k) - \
            self.g_l * (v - self.v_l) - \
            self.g_k_slow * n_slow * (v - self.v_k)
        dn = self.alpha_n(v) * (1.0 - n) - self.beta_n(v) * n
        dh = self.alpha_h(v) * (1.0 - h) - self.beta_h(v) * h
        dn_slow = (self.n_slow_inf(v)-n_slow) / self.tau_n_slow

        return np.array([dv, dn, dh, dn_slow])

    def simulate(self, tspan=None, x0=None):

        x0 = self.set_initial_state() if x0 is None else x0
        tspan = self.tspan if tspan is None else tspan
        sol = odeint(self.f_sys, x0, tspan)

        return {"t": tspan,
                "v":    sol[:, 0],
                "h":    sol[:, 1],
                "n":    sol[:, 2],
                'n_slow': sol[:, 3]
                }
