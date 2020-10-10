import numpy as np
from cvxopt import matrix
from cvxopt import solvers

import numpy as np
import gym
from gym import spaces

import matplotlib.pyplot as plt

# local modules
from cbf.utilities import cbf_2, cbf_3, plot_signals
from models.buck import Buck_Converter_V0


env = Buck_Converter_V0()

env.reset()
print(env.state)
obs = []
N_steps = 10**5
G_diss = 0.0005
print('input: {}'.format(cbf_3(env, dV =  5)))
for i in range(int(2e4)):
    if i%int(1e4) == 0 and i!=0:
        env.G = (1.5)*env.G
        print('step: ', i)
    #print('input: {}'.format(cbf_2(env)))
    u=cbf_3(env, dV =  1.5, eta_1= .5, eta_2=.5)
    #u=cbf_3(env, dV =  1, eta_1= env.R*env.T/env.L, eta_2=env.R*env.T/env.L)
    #print(u)
    s, r, _, _ = env.step(u)
    obs.append(s)

trajectory = np.concatenate(obs).reshape((int(2e4) ,env.observation_space.shape[0]))
plot_signals(trajectory, env.Ides, env.Vdes, dt = 1e-5, dv = 1.5)