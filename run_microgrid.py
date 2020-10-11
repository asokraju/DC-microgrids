import numpy as np
from cvxopt import matrix
from cvxopt import solvers

import numpy as np
import gym
from gym import spaces

import matplotlib.pyplot as plt

# local modules
from cbf.utilities import cbf_microgrid_v0, cbf_microgrid_v1, cbf_microgrid_dist_v0, plot_signals
from models.buck_microgrid import Buck_microgrid_v0, Buck_microgrid_v1


with_L = False

inc_mat = np.array([[-1, 0, 0, -1],[1, -1, 0, 0],[0, 1, -1, 0],[0, 0, 1, 1  ]])
lap_mat = np.matmul(inc_mat, inc_mat.T)
if with_L:
    env = Buck_microgrid_v1(dt = 1e-6)
else:
    env = Buck_microgrid_v0(dt = 1e-6)

theta = np.array([1.,2.,3.,4.])
state = env.reset()
W= np.diag([1.,1.,1.,1.])
print(env.state)
obs = []
N_steps = 10**5

for i in range(int(6e4)):
    u=[]
    theta = theta + env.T*np.dot(np.matmul(lap_mat,W), state[0:4])
    u_dist = np.dot(np.matmul(W, lap_mat),theta)
    for node in range(4):
        if i==int(1e4):
            #env.G = (1.2)*env.G
            print('step: ', i)
    #print('input: {}'.format(cbf_2(env)))
        if with_L:
            u_c = cbf_microgrid_v1(env, node =node, u_dist = u_dist, dV =  1.5, eta_1= .5, eta_2=.5)
            u_net = (u_c-u_dist[node])/env.Vs[node]
            u.append(u_net)
            #u.append(cbf_microgrid_v1(env, node =node, u_dist = u_dist, dV =  3, eta_1= .9, eta_2=.9))
        else:
            u_c = cbf_microgrid_dist_v0(env, node =node, u_dist = u_dist, dV =  20, eta_1= .5, eta_2=.5)
            u_net = (u_c-u_dist[node])/env.Vs[node]
            u.append(u_net)
            #u.append(cbf_microgrid_v0(env, node =node, u_dist = u_dist, dV =  3, eta_1= .9, eta_2=.9))
        #u=cbf_3(env, dV =  1, eta_1= env.R*env.T/env.L, eta_2=env.R*env.T/env.L)
        #print(u)
    state, r, _, _ = env.step(u)
        #obs.append(s)
path = './Power-Converters/DC-microgrids/results/'
#trajectory = np.concatenate(obs).reshape((int(2e4) ,env.observation_space.shape[0]))
#plot_signals(trajectory, env.Ides, env.Vdes, dt = 1e-5, dv = 1.5)
env.plot(savefig_filename = path + 'microgrid_l.png')