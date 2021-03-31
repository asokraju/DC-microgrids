import numpy as np
from cvxopt import matrix
from cvxopt import solvers

import numpy as np
import gym
from gym import spaces
import datetime
from scipy.io import savemat, loadmat

params = {#'legend.fontsize': 'x-large',
          #'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'pdf.fonttype' : 42,
         'ps.fonttype' : 42
         }
import matplotlib.pylab as pylab
import matplotlib
pylab.rcParams.update(params)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

# matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# local modules
from cbf.utilities import cbf_decentralized_2, plot_dutyratio, plot_current, plot_voltage
from models.buck_microgrid import Buck_microgrid_v0, Buck_microgrid_v1


without_L = True
# Percent_load_change
x= 0.13
inc_mat = np.array([[-1, 0, 0, -1],[1, -1, 0, 0],[0, 1, -1, 0],[0, 0, 1, 1  ]])
lap_mat = np.matmul(inc_mat, inc_mat.T)
if without_L:
    env = Buck_microgrid_v1(dt = 10e-6)
else:
    env = Buck_microgrid_v0(dt = 10e-6)

state = env.reset()
W= np.diag([1.,1.,1.,1.])*100
print(env.state)
obs = []
N_steps = 50000
dv = 1
Vs, T, G = env.Vs, env.T, np.diag(env.G)
vl, vh = 229, 231
Il, Ih = vl*G*(1-abs(x)), vh*G*(1+abs(x))
eta_1, eta_2 = .5, .6
delta_1, delta_2 = eta_1*np.diag(env.L)/env.T, eta_2*np.diag(env.L)/env.T
GL, GH = G*(1-abs(x)), G*(1+abs(x))

Il, Ih = vl*GL, vh*GH

for i in range(N_steps):
    u=[]
    I, V = env.state[:4], env.state[4:]
    for node in range(4):
        ui = cbf_decentralized_2(env, node, Il[node], Ih[node], eta_1= 0.9, eta_2=0.9)
        #(I[node], V[node], VL[node], VH[node], Vs[node], delta_1[node], delta_2[node], GL[node], GH[node], T)
        u.append(ui)
    s,_,_,_ = env.step(u)
    if i==int(N_steps*0.4):
        env.G = env.G*(1+x)
        # env.compute_desired()
    print("\r", "Steps = {}/{}".format( i,N_steps), end="")
env.plot()
print(Il, Ih)
data = env.data()
data['Il'] = Il
data['Ih'] = Ih
data['vl'] = vl
data['vh'] = vh
data['a_des'] = env.udes
data['dt'] = env.T
mat_filename = 'data_' + datetime.datetime.now().strftime("%y-%m-%d-%H-%M") + '.mat'
savemat(mat_filename,data)

plot_voltage(data, save_fig=True,savefig_filename='voltage.pdf', FontSize=14)
plot_current(data, save_fig=True,savefig_filename='current.pdf', FontSize=14)
plot_dutyratio(data, save_fig=True,savefig_filename='dutyratio.pdf', FontSize=14)