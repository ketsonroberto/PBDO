# THIS IS A FILE TO TEST THE CODE. DO NOT USE IT AS PART OF THE CODE.

import matplotlib.pyplot as plt
import numpy as np
from StochasticMechanics import Stochastic
from scipy.optimize import minimize
from Performance import PerformanceOpt
from Hazards import Stationary
from Building import *
from BuildingProperties import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import optimize

freq = np.linspace(0.00001, 20, 500)

gamma = np.ones((ndof)) * [0.5]
nu = np.ones((ndof)) * [0.5]
alpha = np.ones((ndof)) * [1]

m = np.ones((ndof)) * [1]
c = np.ones((ndof)) * [1]
k = np.ones((ndof)) * [200]
a = np.ones((ndof)) * [0.8] #0.01
ksi = np.ones((ndof)) * [0.05]
# ksi = [0.05, 0.05]

im_max = 30
B_max = 1

# S1 = np.ones(ndof)
# Ps = Stationary(power_spectrum_object='white_noise', ndof=ndof)
# power_spectrum = Ps.power_spectrum_excitation(freq=freq, S0=S1)

# Von Karman
Ps = Stationary(power_spectrum_object='windpsd', ndof=ndof)
power_spectrum, U = Ps.power_spectrum_excitation(u10=6.2371, freq=freq, z=z)

# plt.semilogy(freq/(2*np.pi), power_spectrum[:,0])
# plt.show()

# columns["area"] = 0.001
# columns.update({"area": 0.001})

ks = []
ms = []
msf = []
#cost = []
nlc = 100
lc = np.linspace(0.05, 2, nlc)

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# fig.suptitle('Mass and Stiffness')
# ax1.plot(lc,ms)
# ax1.plot(lc,msf)
# ax2.plot(lc,ks)
# ax3.plot(ks,cost)
# plt.show()

columns = update_columns(columns=columns, lx=0.4, ly=0.4)

Building = Structure(building, columns, slabs, core, concrete, steel)
k_story = Building.stiffness_story()
m_story = Building.mass_storey(top_story=False)
m_story_f = Building.mass_storey(top_story=True)

k = np.ones(ndof) * [k_story]
m = np.ones(ndof) * [m_story]
m[-1] = m_story_f

length = 0.3
size_col = np.ones(ndof) * [length]

Sto = Stochastic(power_spectrum=power_spectrum, model='bouc_wen', ndof=ndof, freq=freq)

#Opt = PerformanceOpt(power_spectrum=power_spectrum, model='bouc_wen', freq=freq, tol=1e-5, maxiter=100,
#                     design_life=1)  # design_life = 50

# total_cost = Opt.objective_function(size_col=size_col, ksi=ksi, im_max=im_max, B_max=B_max, gamma=gamma, nu=nu,
#                                    alpha=alpha, a=a)

#CostFailure = Costs(building=building, columns=columns, slabs=slabs, core=core, concrete=concrete,
#                    steel=steel, cost=cost)
#size_col = np.ones(ndof) * [0.5]

#size_col = np.array([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
#size_col = np.array([0.1, 0.2, 0.3])

args=[ksi, im_max, B_max, gamma, nu, alpha, a]
sizea = 0.1
sizeb = 1
wa = 0.1
wb=100

npar = 10
nw = 10
X = np.zeros((npar * nw, 3 * ndof + 1))
y = np.zeros((npar * nw, 2 * ndof))
ct=0
ct1=0
for kk in range(npar):
    size_col = sizea+(sizeb-sizea)*np.random.rand(ndof)

    M, C, K, m, c, k = Sto.get_MCK(size_col=size_col, args=args, columns=columns)

    for i in range(nw):
        im = wa + (wb - wa) * np.random.rand(1)[0]

        idd = 0
        for j in np.arange(0, 3 * ndof, 3):
            X[ct, j] = m[idd]
            X[ct, j + 1] = c[idd]
            X[ct, j + 2] = k[idd]
            idd = idd + 1

        X[ct, -1] = im
        ct = ct + 1

        Ps = Stationary(power_spectrum_object='windpsd', ndof=ndof)
        power_spectrum, ub = Ps.power_spectrum_excitation(u10=im, freq=freq, z=z)
        Var, Vard = Sto.statistical_linearization(M=M, C=C, K=K, power_sp=power_spectrum, tol=0.01, maxiter=100,
                                                  gamma=gamma, nu=nu, alpha=alpha, a=a)


        idd = 0
        for j in np.arange(0, 2 * ndof, 2):
            y[ct1, j] = Var[idd][0]
            y[ct1, j + 1] = Vard[idd][0]
            idd = idd + 1

        ct1 = ct1 + 1


print(np.shape(y))

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)

kernels_U = [None,
             ConstantKernel(1.0, (1e-4, 1e4)) * RBF(1, (1e-4, 1e4)),
           1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
           1.0 * ExpSineSquared(length_scale=1.0, periodicity=1,
                                length_scale_bounds=(1.0e-5, 100.0),
                                periodicity_bounds=(1.0, 10.0)),
           ConstantKernel(0.1, (0.01, 10.0))
           * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.0, 10.0)) ** 2),
           1.0 * Matern(length_scale=1.0, nu=1.5)]

gp = GaussianProcessRegressor(kernel=kernels_U[0], n_restarts_optimizer=10, normalize_y=False)
gp.fit(X, y)

r2 = gp.score(X, y)
print(r2)


yp = gp.predict(np.array(X[2].reshape(1, -1)))

val = X[2]
val[-1]=100.0
print(val)
yp = gp.predict(val.reshape(1, -1))

print(yp)

#print(np.shape(X))
#print(np.shape(y))

#nn_architecture = [
#    {"input_dim": 10, "output_dim": 25, "activation": "relu"},
#    {"input_dim": 25, "output_dim": 50, "activation": "relu"},
#    {"input_dim": 50, "output_dim": 50, "activation": "relu"},
#    {"input_dim": 50, "output_dim": 25, "activation": "relu"},
#    {"input_dim": 25, "output_dim": 6, "activation": "relu"},
#]

#from neural import NeuralNets
#from sklearn.model_selection import train_test_split

#NN = NeuralNets(nn_architecture)

#TEST_SIZE = 0.1
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=132)

##print(X_train)
#params_values, cost_history = NN.train(X=np.transpose(X_train), Y=np.transpose(y_train), epochs=1000,
#                                       learning_rate=1, verbose=True)

"""
b0 = np.linspace(0.1, 0.5, 20)
cost_f = []
cost_i = []
cost_t = []
mm = []
pp = []
args=[ksi, im_max, B_max, gamma, nu, alpha, a]
for i in range(len(b0)):
    Cf = CostFailure.cost_damage(b=b0[i], col_size=size_col[0], L=columns["height"], ncolumns=columns["quantity"],
                                 dry_wall_area=dry_wall_area)
    Ci = CostFailure.initial_cost_stiffness(col_size=b0[i], par0=25.55133, par1=0.33127)

    scol = np.array([b0[i], b0[i]])
    Ct = Opt.objective_function(size_col=scol, args=args)
    #mom, phi = Building.compression(col_size=b0[i], L=columns["height"])

    cost_f.append(Cf)
    cost_i.append(Ci)
    cost_t.append(Ct)

fig = plt.figure()
plt.plot(b0, cost_t,'-o')
plt.show()

#fig = plt.figure()
#plt.plot(phi, mom,'-o')
#plt.show()
"""
"""
b0 = np.linspace(0.05,0.5,5)
b1 = np.linspace(0.05,0.5,5)
B0, B1 = np.meshgrid(b0, b1)
args=[ksi, im_max, B_max, gamma, nu, alpha, a]
tc = np.zeros((5, 5))
for i in range(len(b0)):
    print(i)
    for j in range(len(b1)):
        size_col = np.array([b0[i], b1[j]])
        resp = Opt.objective_function(size_col=size_col, args=args)
        tc[i,j] = resp


Z = tc.reshape(B0.shape)

Z = np.array(Z)
nd = np.unravel_index(np.argmin(Z, axis=None), Z.shape)
print([B0[nd], B1[nd]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(B0, B1, np.log(Z), cmap=plt.cm.get_cmap('plasma'),linewidth=0, antialiased=False)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
"""
#size_col = np.ones(ndof) * [0.2]
#args=[ksi, im_max, B_max, gamma, nu, alpha, a]
##args = {"ksi": ksi, "im_max": im_max, "B_max": B_max, "gamma": gamma, "nu": nu, "alpha": alpha, "a": a}
#bnds = []
#for i in range(ndof):
#    bnds.append((0.1, 1))
#bnds=tuple(bnds)

###from scipy import optimize
###res = optimize.fmin(Opt.objective_function, x0=size_col)


#res = minimize(Opt.objective_function, x0=size_col, args=args, bounds=bnds)

###from scipy.optimize import basinhopping
###minimizer_kwargs = {"method": "BFGS", "args": args}
###ret = basinhopping(Opt.objective_function, x0=size_col, minimizer_kwargs=minimizer_kwargs, niter=200)

#print(res)

### Global methods.
###from scipy.optimize import rosen, shgo
###from scipy.optimize import dual_annealing
###ret = dual_annealing(Opt.objective_function, bounds=bnds)
###print((ret.x, ret.fun))

#c = Opt.linear_damping(m=m, k=k, ksi=ksi)
#M, C, K = Opt.create_mck(m=m, c=c, k=k, gamma=gamma, nu=nu, alpha=alpha, a=a)
#financial_loss_rate = Opt.stochastic_financial_loss(M=M, C=C, K=K, stiff=k, im_max=im_max,
#                                                B_max=B_max, size_col=size_col, Nim=1, NB=1, gamma=gamma, nu=nu,
#                                                alpha=alpha, a=a)

