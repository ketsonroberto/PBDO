import numpy as np
from numpy.linalg import inv
import scipy as sp
from os import path
import copy
import sys
from Building import *
from BuildingProperties import *

from Hazards import Stationary

class Stochastic:

    def __init__(self, power_spectrum=None, model=None, ndof=None, freq=None):

        # todo: implement the case of full matrix power spectrum
        self.model = model
        self.ndof = ndof
        self.freq = freq
        self.power_spectrum = power_spectrum

        # Initial guess of the iterative process in the statistical linearization
        if ndof >= 1:
            self.meq0 = np.ones(ndof) * [0]
            self.ceq0 = np.ones(ndof) * [0.5]
            self.keq0 = np.ones(ndof) * [1e3]
        else:
            raise ValueError('ndof MUST be larger than or equal to 1.')

        if self.model is 'bouc_wen':
            self.create_matrices = 'create_matrix_bw'
            self.equivalent_elements = 'equivalent_elements_bw'
            self.mck_matrices = 'mck_matrices_bw'

    def statistical_linearization(self, M=None, C=None, K=None, power_sp=None, tol=1e-3, maxiter=1000, **kwargs):

        if tol<sys.float_info.min:
            raise ValueError('tol cannot be lower than '+str(sys.float_info.min))

        if not isinstance(maxiter,int):
            raise TypeError('maxiter MUST be an integer. ')

        if maxiter<1:
            raise ValueError('maxiter cannot be lower than 1')

        freq = self.freq
        #power_spectrum = self.power_spectrum
        power_spectrum = power_sp
        Mt = np.zeros(np.shape(M))
        Ct = np.zeros(np.shape(C))
        Kt = np.zeros(np.shape(K))

        meq = copy.copy(self.meq0)
        ceq = copy.copy(self.ceq0)
        keq = copy.copy(self.keq0)

        #meq_1 = copy.copy(self.meq0)/100
        #ceq_1 = copy.copy(self.ceq0)/100
        #keq_1 = copy.copy(self.keq0)/100
        meq_1 = 1e-3
        ceq_1 = 1e-3
        keq_1 = 1e-3

        # Do not use mass as a stop criterion

        Meq, Ceq, Keq = Stochastic.assembly_matrices(self, ceq, keq)
        runs = 1
        error = 1000*tol
        while error > tol and runs <= maxiter:

            Mt = M + Meq
            Ct = C + Ceq
            Kt = K + Keq

            H = []
            for i in range(len(freq)):
                H.append(inv(-(freq[i]**2) * Mt + 1j * freq[i] * Ct + Kt))

            ceq_1 = copy.copy(ceq)
            keq_1 = copy.copy(keq)
            ceq, keq, Var, Vard = Stochastic.update_matrices(self, H, power_spectrum, ceq, keq, kwargs)
            Meq, Ceq, Keq = Stochastic.assembly_matrices(self, ceq, keq)

            dceq = abs(ceq - ceq_1) / abs(ceq_1)
            dkeq = abs(keq - keq_1) / abs(keq_1)

            error = np.min(np.minimum(dceq, dkeq))
            runs = runs + 1

            return Var, Vard

    def assembly_matrices(self, ceq=None, keq=None):

        assembly_matrices_fun = eval("Stochastic." + self.create_matrices)
        Meq, Ceq, Keq = assembly_matrices_fun(self, ceq, keq)

        return Meq, Ceq, Keq

    def create_matrix_bw(self, ceq=None, keq=None):

        ndof = self.ndof
        Meq = np.zeros((2 * ndof, 2 * ndof))
        Ceq = np.zeros((2 * ndof, 2 * ndof))
        Keq = np.zeros((2 * ndof, 2 * ndof))

        cont = 0
        for i in np.arange(ndof,2 * ndof):
            Ceq[i, cont] = ceq[cont]
            Keq[i, i] = keq[cont]
            cont = cont + 1

        return Meq, Ceq, Keq

    def update_matrices(self, H=None, power_spectrum=None, ceq=None, keq=None, kwargs=None):

        update_matrices_fun = eval("Stochastic." + self.equivalent_elements)
        ceq, keq, Var, Vard = update_matrices_fun(self, H, power_spectrum, ceq, keq, kwargs)

        return ceq, keq, Var, Vard

    def equivalent_elements_bw(self, H=None, power_spectrum=None, ceq_in=None, keq_in=None, kwargs=None):

        if 'gamma' in kwargs.keys():
            gamma = kwargs['gamma']
        else:
            raise ValueError('gamma cannot be None.')

        if 'nu' in kwargs.keys():
            nu = kwargs['nu']
        else:
            raise ValueError('nu cannot be None.')

        if 'alpha' in kwargs.keys():
            alpha = kwargs['alpha']
        else:
            raise ValueError('alpha cannot be None.')

        H_ps = []
        H_ps_freq = []
        aux = np.zeros((len(self.freq), 2*self.ndof))
        for i in range(len(self.freq)):
            aux0 = np.sum(abs(H[i][:,0:self.ndof])**2, axis=1)
            aux[i, 0:self.ndof] = 2*np.diag(power_spectrum[i]).dot(aux0[0:self.ndof])
            aux[i, self.ndof:2*self.ndof] = 2*np.diag(power_spectrum[i]).dot(aux0[self.ndof:2*self.ndof])
            H_ps.append(aux[i])
            H_ps_freq.append((self.freq[i]**2)*aux[i])

        H_ps = np.array(H_ps).T
        H_ps_freq = np.array(H_ps_freq).T

        keq = np.zeros((self.ndof,1))
        ceq = np.zeros((self.ndof, 1))
        Var = np.zeros((self.ndof, 1))
        Vard = np.zeros((self.ndof, 1))
        for i in range(self.ndof):
            Ex = np.trapz(H_ps[i],self.freq)
            Exd = np.trapz(H_ps_freq[i], self.freq)
            Ez = np.trapz(H_ps[i + self.ndof], self.freq)
            Ezd = -(keq_in[i] / ceq_in[i]) * Ez

            Var[i] = Ex
            Vard[i] = Exd
            keq[i] = np.sqrt(2.0/np.pi) * (gamma[i] * np.sqrt(Exd) + nu[i] * Ezd / np.sqrt(Ez))
            ceq[i] = np.sqrt(2.0/np.pi) * (gamma[i] * Ezd / np.sqrt(Exd) + nu[i] * np.sqrt(Ez)) - alpha[i]

        return ceq, keq, Var, Vard

    def create_mck(self, m=None, c=None, k=None, **kwargs):

        mck_matrices_fun = eval("Stochastic." + self.mck_matrices)
        M, C, K = mck_matrices_fun(self, m=m, c=c, k=k, kwargs=kwargs)

        return M, C, K

    def mck_matrices_bw(self, m=None, c=None, k=None, kwargs=None):

        ndof = self.ndof

        if 'gamma' in kwargs.keys():
            gamma = kwargs['gamma']
        else:
            raise ValueError('gamma cannot be None.')

        if 'nu' in kwargs.keys():
            nu = kwargs['nu']
        else:
            raise ValueError('nu cannot be None.')

        if 'alpha' in kwargs.keys():
            alpha = kwargs['alpha']
        else:
            raise ValueError('alpha cannot be None.')

        if 'a' in kwargs.keys():
            a = kwargs['a']
            for i in range(ndof):
                if a[i] < 0:
                    raise ValueError('a cannot be less than 0.')
        else:
            raise ValueError('a cannot be None.')

        M = np.zeros((2 * ndof, 2 * ndof))
        C = np.eye(2 * ndof)
        K = np.zeros((2 * ndof, 2 * ndof))
        # Mass matrix

        for i in range(ndof):
            C[i,i] = c[i]
            K[i,i] = a[i]*k[i]
            K[i, i+ndof] = (1-a[i]) * k[i]

            if i < ndof-1:
                C[i,i+1] = -c[i]
                K[i, i + 1] = -a[i]*k[i]
                K[i, i + ndof + 1] = -(1 - a[i]) * k[i]

            for j in range(i+1):
                M[i,j] = m[i]

        return M, C, K

    def linear_damping(self, m=None, k=None, ksi=None):

        ndof = self.ndof

        Minv = np.zeros((ndof, ndof))
        M = np.zeros((ndof, ndof))
        MK = np.zeros((ndof, ndof))
        C = np.zeros((ndof, ndof))
        K = np.zeros((ndof, ndof))
        c = np.zeros((ndof))

        for i in range(ndof):
            if m[i] <= 0:
                raise ValueError('Mass cannot be lower than or equal to zero!')
            Minv[i,i] = 1/m[i]
            M[i,i] = m[i]

            if i < ndof-1:
                K[i, i] = k[i] + k[i + 1]
                K[i, i + 1] = -k[i + 1]
                K[i + 1, i] = -k[i + 1]

        K[-1,-1] = k[-1]
        MK = Minv.dot(K)

        wn = np.linalg.eigvals(MK)
        wn = np.sqrt(wn)
        # Damping proportional to mass and stiffness such that: [C] = b0 * [M] + b1 * [K]

        b0 = ((2 * wn[1] * wn[0]) / (wn[0] ** 2 - wn[1] ** 2))*(wn[0] * ksi[1] - wn[1] * ksi[0])
        b1 = (2 / (wn[0] ** 2 - wn[1] ** 2))*(wn[0] * ksi[0] - wn[1] * ksi[1])

        # Damping proportional to stiffness
        C = b1 * K

        for i in range(ndof-1):
            c[i+1] = -C[i,i+1]

        c[0] = C[0,0]-c[1]
        return c

    @staticmethod
    def linear_mean_response(k=None, F=None, a=None):

        ndof = len(k)
        KF = np.zeros((ndof, ndof))
        K = np.zeros((ndof, ndof))
        w = np.zeros((ndof))
      
        for i in range(ndof):
            K[i,i] = a[i]*k[i]
            w[i] = F[i]

            if i < ndof-1:
                K[i, i + 1] = -a[i]*k[i]

        KF = np.linalg.inv(K)
        mean_response = KF.dot(F)

        return mean_response

    def get_MCK(self, size_col=None, args=None, columns=None):
        # im_max: maximum intensity measure
        # B_max: maximum barrier level
        # ndof = 2
        # im_max = 30
        # B_max = 1
        # ksi = np.ones((ndof)) * [0.05]
        # gamma = np.ones((ndof)) * [0.5]
        # nu = np.ones((ndof)) * [0.5]
        # alpha = np.ones((ndof)) * [1]
        # a = np.ones((ndof)) * [1.0]  # 0.01

        ksi = args[0]
        im_max = args[1]
        B_max = args[2]
        gamma = args[3]
        nu = args[4]
        alpha = args[5]
        a = args[6]

        #ndof = self.ndof
        columns = update_columns(columns=columns, lx=size_col, ly=size_col)
        Building = Structure(building, columns, slabs, core, concrete, steel, cost)

        if len(size_col) != ndof:
            raise ValueError('length of size_col is not equal to ndof!')

        initial_cost = 0
        k = []
        for i in range(ndof):
            self.columns = update_columns(columns=columns, lx=size_col[i], ly=size_col[i])
            Ix = size_col[i] ** 4 / 12
            Iy = Ix  # Square section
            area = size_col[i] ** 2  # square section

            Building = Structure(building, columns, slabs, core, concrete, steel, cost)
            Cost = Costs(building, columns, slabs, core, concrete, steel, cost)
            stiffness = Building.stiffness_story()
            k.append(stiffness)

            initial_cost = initial_cost + Cost.initial_cost_stiffness(col_size=size_col[i], par0=25.55133, par1=0.33127)
        # k[end] top floor
        k = np.array(k)
        mass = Building.mass_storey(top_story=False)
        mass_top = Building.mass_storey(top_story=True)
        m = np.ones((ndof)) * [mass]
        m[-1] = mass_top  # Top floor is m[end] - include water reservoir
        # Estimate the damping.

        c = self.linear_damping(m=m, k=k, ksi=ksi)
        M, C, K = self.create_mck(m=m, c=c, k=k, gamma=gamma, nu=nu, alpha=alpha, a=a)

        return M, C, K, m, c, k