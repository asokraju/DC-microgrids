import numpy as np
from cvxopt import matrix
from cvxopt import solvers

import numpy as np
import gym
from gym import spaces

import matplotlib.pyplot as plt

# local modules
from cbf.utilities import cbf_microgrid, plot_signals
from models.buck_microgrid import Buck_microgrid_v0


env = Buck_microgrid_v0()

env.reset()
print(env.state)
obs = []
N_steps = 10**5
G_diss = 0.0005
#print('input: {}'.format(cbf_3(env, dV =  5)))
for i in range(int(2e4)):
    u=[]
    for node in range(4):
        if i%int(1e4) == 0 and i!=0:
            env.G = (1.5)*env.G
            print('step: ', i)
    #print('input: {}'.format(cbf_2(env)))
        u.append(cbf_microgrid(env, node =node, dV =  1.5, eta_1= .5, eta_2=.5))
        #u=cbf_3(env, dV =  1, eta_1= env.R*env.T/env.L, eta_2=env.R*env.T/env.L)
        #print(u)
    s, r, _, _ = env.step(u)
        #obs.append(s)

#trajectory = np.concatenate(obs).reshape((int(2e4) ,env.observation_space.shape[0]))
#plot_signals(trajectory, env.Ides, env.Vdes, dt = 1e-5, dv = 1.5)
env.plot()