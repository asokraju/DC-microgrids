import numpy as np
from cvxopt import matrix
from cvxopt import solvers

import numpy as np
import gym
from gym import spaces

import matplotlib.pyplot as plt

# local modules
from cbf.utilities import cbf_microgrid_v0, cbf_microgrid_v1, plot_signals
from models.buck_microgrid import Buck_microgrid_v0, Buck_microgrid_v1

with_L = False
if with_L:
    env = Buck_microgrid_v1()
else:
    env = Buck_microgrid_v0()

env.reset()
print(env.state)
obs = []
N_steps = 10**5
G_diss = 0.0005
#print('input: {}'.format(cbf_3(env, dV =  5)))
for i in range(int(6e4)):
    u=[]
    for node in range(4):
        if i==int(1e4):
            env.G = (1.2)*env.G
            print('step: ', i)
    #print('input: {}'.format(cbf_2(env)))
        if with_L:
            u.append(cbf_microgrid_v1(env, node =node, dV =  1.5, eta_1= .5, eta_2=.5))
        else:
            u.append(cbf_microgrid_v0(env, node =node, dV =  1.5, eta_1= .5, eta_2=.5))
        #u=cbf_3(env, dV =  1, eta_1= env.R*env.T/env.L, eta_2=env.R*env.T/env.L)
        #print(u)
    s, r, _, _ = env.step(u)
        #obs.append(s)
path = './Power-Converters/DC-microgrids/results/'
#trajectory = np.concatenate(obs).reshape((int(2e4) ,env.observation_space.shape[0]))
#plot_signals(trajectory, env.Ides, env.Vdes, dt = 1e-5, dv = 1.5)
env.plot(savefig_filename = path + 'microgrid_l.png')