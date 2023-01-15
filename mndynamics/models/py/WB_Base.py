import numpy as np
from numpy import exp
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mndynamics.utility import is_sequence


class WB(object):

    def __init__(self, par={}) -> None:

        self.check_parameters(par)
        self.set_parameters(par)

    def __call__(self) -> None:
        print("Wang-Buzsaki Model of an Inhibitory Interneuron in Rat Hippocampus")
        return self._par

    def __str__(self) -> str:
        return "Wang-Buzsaki Model of an Inhibitory Interneuron in Rat Hippocampus"

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
            'g_k': 9.0,
            'g_na': 35.0,
            'g_l': 0.1,
            'v_k': -90.0,
            'v_na': 55.0,
            'v_l': -65.0,
            'i_ext': 0.75,
            't_end': 100.0,
            'v_thr': -55.0,
            'v0': -63.0,
            'dt': 0.01
        }
        return params

    def set_initial_state(self):

        x0 = [self.v0,
              self.h_inf(self.v0),
              self.n_inf(self.v0),
              ]
        return x0

    def alpha_h(self, v):
        return 0.35 * exp(-(v + 58.0) / 20.0)

    def alpha_m(self, v):
        return 0.1 * (v + 35.0) / (1.0 - exp(-0.1 * (v + 35.0)))

    def alpha_n(self, v):
        return -0.05 * (v + 34.0) / (exp(-0.1 * (v + 34.0)) - 1.0)

    def beta_h(self, v):
        return 5.0 / (exp(-0.1 * (v + 28.0)) + 1.0)

    def beta_m(self, v):
        return 4.0 * exp(-(v + 60.0) / 18.0)

    def beta_n(self, v):
        return 0.625 * exp(-(v + 44.0) / 80.0)

    def h_inf(self, v):
        return self.alpha_h(v) / (self.alpha_h(v) + self.beta_h(v))

    def m_inf(self, v):
        return self.alpha_m(v) / (self.alpha_m(v) + self.beta_m(v))

    def n_inf(self, v):
        return self.alpha_n(v) / (self.alpha_n(v) + self.beta_n(v))

    def f_sys(self, x0, t):

        v, h, n = x0

        I_Na = self.g_na * self.m_inf(v) ** 3 * h * (v - self.v_na)
        I_L = self.g_l * (v - self.v_l)
        I_K = self.g_k * n ** 4 * (v - self.v_k)

        dv = -I_Na - I_K - I_L + self.i_ext
        dh = self.alpha_h(v) * (1-h) - self.beta_h(v) * h
        dn = self.alpha_n(v) * (1-n) - self.beta_n(v) * n

        return [dv, dh, dn]

    def simulate(self, tspan=None, x0=None):

        x0 = self.set_initial_state() if x0 is None else x0
        tspan = self.tspan if tspan is None else tspan
        sol = odeint(self.f_sys, x0, tspan)

        return {"t": tspan,
                "v":    sol[:, 0],
                "h":    sol[:, 1],
                "n":    sol[:, 2]
                }


class WB_F_I_CURVE(WB):

    def __init__(self, par={}) -> None:

        super().__init__(par)

    def __call__(self) -> None:
        print("F-I Curve Wang-Buzsaki Model")
        return self._par

    def __str__(self) -> str:
        return "F-I Curve Wang-Buzsaki Model"

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


class WB_Net_Gap(object):

    def __init__(self, par: dict = {}) -> None:
        self.check_parameters(par)
        self.set_parameters(par)

    def __str__(self) -> str:
        return "Wang-Buzsaki Model on Network with Gap junctions"

    def __call__(self) -> None:
        print("Wang-Buzsaki Model on Network with Gap junctions")
        return self._par

    def set_initial_state(self):

        v = np.array(self.v0)
        x0 = np.hstack((v,
                        self.h_inf(v),
                        self.n_inf(v)))
        return x0

    def set_parameters(self, par={}):

        self._par = self.get_default_parameters()
        self._par.update(par)

        for key in self._par.keys():
            setattr(self, key, self._par[key])
        self.tspan = np.arange(0.0, self.t_end, self.dt)

        assert(self.adj is not None), "adjacency matrix is not set"
        self.N = self.adj.shape[0]

        if not is_sequence(self.i_ext):
            self.i_ext = [self.i_ext] * self.N
        if not is_sequence(self.v0):
            self.v0 = [self.v0] * self.N

        self._par['N'] = self.N
        self._par['i_ext'] = self.i_ext
        self._par['v0'] = self.v0

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
            'g_k': 9.0,
            'g_na': 35.0,
            'g_l': 0.1,
            'v_k': -90.0,
            'v_na': 55.0,
            'v_l': -65.0,
            'i_ext': 0.0,
            't_end': 200.0,
            'v_thr': -55.0,
            'v0': -63.0,
            'dt': 0.01,
            'g_gap': 0.01,
            'adj': None,
            'N': None,
        }
        return params

    def beta_h(self, v):
        return 5.0 / (exp(-0.1 * (v + 28.0)) + 1.0)

    def beta_n(self, v):
        return 0.625 * exp(-(v + 44.0) / 80.0)

    def beta_m(self, v):
        return 4.0 * exp(-(v + 60.0) / 18.0)

    def alpha_h(self, v):
        return 0.35 * exp(-(v + 58.0) / 20.0)

    def alpha_m(self, v):
        return 0.1 * (v + 35.0) / (1.0 - exp(-(v + 35.0) / 10.0))

    def alpha_n(self, v):
        return 0.05 * (v + 34.0) / (1.0 - exp(-0.1 * (v + 34.0)))

    def h_inf(self, v):
        return self.alpha_h(v) / (self.alpha_h(v) + self.beta_h(v))

    def m_inf(self, v):
        return self.alpha_m(v) / (self.alpha_m(v) + self.beta_m(v))

    def n_inf(self, v):
        return self.alpha_n(v) / (self.alpha_n(v) + self.beta_n(v))

    def f_sys(self, x0, t):

        N = self.N
        df = np.zeros(3 * N)

        v = x0[:N]
        h = x0[N: 2 * N]
        n = x0[2 * N:]
        m = np.zeros(N)
        I_syn = np.zeros(N)

        for i in range(N):

            m[i] = self.alpha_m(
                v[i]) / (self.alpha_m(v[i]) + self.beta_m(v[i]))
            I_Na = self.g_na * m[i] ** 3 * h[i] * (v[i] - self.v_na)
            I_L = self.g_l * (v[i] - self.v_l)
            I_K = self.g_k * n[i] ** 4 * (v[i] - self.v_k)

            for j in range(N):
                if i != j:
                    I_syn[i] += (v[j] - v[i])

            I_syn[i] *= self.g_gap
            df[i] = -I_Na - I_K - I_L + I_syn[i] + \
                self.i_ext[i]                                   # dv
            df[i + N] = self.alpha_h(v[i]) * (1-h[i]) - \
                self.beta_h(v[i]) * h[i]                        # dh
            df[i + 2 * N] = self.alpha_n(v[i]) * (1 - n[i]) - \
                self.beta_n(v[i]) * n[i]                        # dn

        return df

    def simulate(self, tspan=None, x0=None):

        x0 = self.set_initial_state() if x0 is None else x0
        tspan = self.tspan if tspan is None else tspan
        sol = odeint(self.f_sys, x0, tspan)

        N = self.N

        return {"t": tspan,
                "v":    sol[:, 0: N],
                "h":    sol[:, N: 2 * N],
                "n":    sol[:, 2 * N: 3 * N],
                }
