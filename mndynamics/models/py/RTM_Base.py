import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import odeint
from mndynamics.utility import is_sequence


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

    def simulate(self, tspan=None, x0=None):

        x0 = self.set_initial_state() if x0 is None else x0
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

    def simulate(self, tspan=None, x0=None):
        """
        simulate the model

        Parameters
        ----------
        tspan : array_like
            time span
        x0 : array_like
            initial state

        Returns
        -------
        dict : 
            time series of membrane potential (v), activation variables 
            m, h, n and w
        """

        x0 = self.set_initial_state() if x0 is None else x0
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

    def simulate(self, tspan=None, x0=None):

        x0 = self.set_initial_state() if x0 is None else x0
        tspan = self.tspan if tspan is None else tspan
        sol = odeint(self.f_sys, x0, tspan)

        return {"t": tspan,
                "v":    sol[:, 0],
                "m":    self.m_inf(sol[:, 0]),
                "h":    sol[:, 1],
                "n":    sol[:, 2],
                "ca":    sol[:, 3]
                }


class RTM_2D(RTM):

    '''
    Reduced 2-dimensional Traub-Miles Model of a Pyramidal Neuron in Rat Hippocampus
    as defined in Eq. 12.1-2 in [1]

    References
    ----------
    [1] Börgers, C., 2017. An introduction to modeling neuronal dynamics (Vol. 66). Berlin: Springer.
    '''

    def __init__(self, par: dict = {}) -> None:

        self.check_parameters(par)
        self.set_parameters(par)

    def __call__(self) -> None:
        print("Reduced 2-dimensional Traub-Miles Model of a Pyramidal Neuron in Rat Hippocampus")
        return self._par

    def __str__(self) -> str:
        return "Reduced 2-dimensional Traub-Miles Model of a Pyramidal Neuron in Rat Hippocampus"

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
              self.n_inf(self.v0)]
        return x0

    def f_sys(self, x0, t):
        '''
        define Reduced 2D RTM Model
        '''
        v, n = x0
        m_inf = self.m_inf(v)
        h = 1.0 - n

        dv = (self.i_ext - self.g_na * m_inf**3 * h * (v - self.v_na) -
              self.g_k * n**4 * (v - self.v_k) - self.g_l * (v - self.v_l)) / self.c

        dn = self.alpha_n(v) * (1.0 - n) - self.beta_n(v) * n
        return np.asarray([dv, dn])

    def n_inf_p(self, v):
        n_inf_p = (self.alpha_n(v)+self.beta_n(v))*self.alpha_n_p(v)
        n_inf_p = n_inf_p-self.alpha_n(v)*(self.alpha_n_p(v)+self.beta_n_p(v))
        n_inf_p = n_inf_p/((self.alpha_n(v)+self.beta_n(v)))**2
        return n_inf_p

    def m_inf_p(self, v):
        m_inf_p = (self.alpha_m(v)+self.beta_m(v))*self.alpha_m_p(v)
        m_inf_p = m_inf_p-self.alpha_m(v)*(self.alpha_m_p(v)+self.beta_m_p(v))
        m_inf_p = m_inf_p/((self.alpha_m(v)+self.beta_m(v)))**2
        return m_inf_p

    def beta_n_p(self, v):
        beta_n = 0.5*exp(-(v+57)/40)
        beta_n_p = -beta_n/40
        return beta_n_p

    def beta_m_p(self, v):
        num = 0.28*(v+27)
        den = exp((v+27)/5)-1
        num_p = 0.28
        den_p = exp((v+27)/5)/5
        beta_m_p = (den*num_p-num*den_p)/(den**2)
        return beta_m_p

    def alpha_n_p(self, v):
        num = 0.032*(v+52)
        den = 1-exp(-(v+52)/5)
        num_p = 0.032
        den_p = exp(-(v+52)/5)/5
        alpha_n_p = (den*num_p-num*den_p)/(den**2)
        return alpha_n_p

    def alpha_m_p(self, v):
        num = 0.32*(v+54)
        den = 1-exp(-(v+54)/4)
        num_p = 0.32
        den_p = exp(-(v+54)/4)/4
        alpha_m_p = (den*num_p-num*den_p)/(den**2)
        return alpha_m_p

    def jacobian(self, x0):
        '''
        Jacobian of the Reduced 2D RTM Model
        '''
        v, n = x0
        m_inf = self.m_inf(v)
        m_inf_p = self.m_inf_p(v)
        beta_n_p = self.beta_n_p(v)
        alpha_n_p = self.alpha_n_p(v)
        alpha_n = self.alpha_n(v)
        beta_n = self.beta_n(v)

        h = 1.0 - n

        g_na = self.g_na
        g_k = self.g_k
        g_l = self.g_l
        v_na = self.v_na
        v_k = self.v_k
        v_l = self.v_l
        inv_c = 1.0/self.c

        J = np.zeros((2, 2), dtype=float)
        J[0, 0] = (g_na*3*m_inf**2*m_inf_p*(1-n)*(v_na-v) -
                   g_na*m_inf**3*(1-n) -
                   g_k*n**4-g_l) * inv_c
        J[0, 1] = (-g_na*m_inf**3*(v_na-v)+g_k*4*n**3*(v_k-v))*inv_c
        J[1, 0] = (alpha_n_p*(1-n)-beta_n_p*n)*inv_c
        J[1, 1] = (-alpha_n-beta_n)*inv_c

        return J

    def F(self, v, i_ext):
        g_na = self.g_na
        g_k = self.g_k
        g_l = self.g_l
        v_na = self.v_na
        v_k = self.v_k
        v_l = self.v_l
        m_inf = self.m_inf(v)
        n_inf = self.n_inf(v)

        return (g_na*m_inf**3*(1-n_inf)*(v_na-v) +
                g_k*n_inf**4*(v_k-v) +
                g_l*(v_l-v)+i_ext)

    def F2(self, v, i_ext):
        return self.F(v, i_ext)**2

    def classify_eig(self, E):
        '''
        Classify the eigenvalues of the Jacobian
        both real and negative: stable
        both real and positive: unstable
        complex conjugate pair, negative real part: stable spiral
        complex conjugate pair, positive real part: unstable spiral
        both real, one positive, one negative: saddle
        '''
        assert(len(E) == 2), 'length of input should be 2'
        E = np.array(E)

        e1, e2 = E
        # check if both are real
        if np.all(e1.imag < 1e-10) and np.all(e2.imag < 1e-10):
            if np.all(e1.real < 0) and np.all(e2.real < 0):
                return 'stable'
            elif np.all(e1.real > 0) and np.all(e2.real > 0):
                return 'unstable'
            else:
                return 'saddle'
        else:
            if self._if_complex_conjugate(e1, e2):
                if np.all(e1.real < 0):
                    return 'stable spiral'
                elif np.all(e1.real > 0):
                    return 'unstable spiral'
                else:
                    return 'unknown'

    def _if_complex_conjugate(self, a, b):
        '''
        check if 2 input are complex conjugate pair
        '''
        if np.all(a.real == b.real) and np.all(a.imag == -b.imag):
            return True
        else:
            return False

    def simulate_bif(self, i_ext_vec):
        '''
        Simulate the bifurcation diagram
        '''
        v_k = self.v_k
        v_na = self.v_na
        v_l = self.v_l
        g_l = self.g_l

        stability = {
            'stable': [],
            'unstable': [],
            'saddle': [],
            'stable spiral': [],
            'unstable spiral': []
        }

        assert(is_sequence(i_ext_vec)), 'i_ext should be a sequence'
        for i in range(len(i_ext_vec)):
            i_ext = i_ext_vec[i]
            v_min = min(v_k, v_l+i_ext/g_l)
            v_max = max(v_na, v_l+i_ext/g_l)
            v_vec = np.linspace(v_min, v_max, 100)
            v_vec_left = v_vec[:-1]
            v_vec_right = v_vec[1:]
            f2 = self.F(v_vec_left, i_ext) * self.F(v_vec_right, i_ext)
            ind = np.where(f2 < 0)[0]

            for j in range(len(ind)):
                v_root = brentq(self.F,
                                v_vec_left[ind[j]],
                                v_vec_right[ind[j]],
                                args=(i_ext,))
                n_root = self.n_inf(v_root)
                J = self.jacobian([v_root, n_root])
                E = np.linalg.eigvals(J)
                stability[self.classify_eig(E)].append([i_ext, v_root])

        return stability

    def plot_bif(self, stability, ax=None):
        '''
        Plot the bifurcation diagram
        '''
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.set_xlabel(r'$\mu A/cm^2$')
        ax.set_ylabel(r'$v^{*}$ [mV]')

        for key in stability.keys():
            if len(stability[key]) > 0:
                stability[key] = np.array(stability[key])
                ax.plot(stability[key][:, 0], stability[key]
                        [:, 1], 'o', label=key, markersize=4, alpha=0.7)
        ax.legend(frameon=False, loc='best')
