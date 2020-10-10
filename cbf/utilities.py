import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import matplotlib.pyplot as plt

#Build barrier function model
def cbf_2(env, eta_1 = 0.5, eta_2 = 0.5, dV = 5, G_diss = 0.02):
    eta_1 = 0.5 
    eta_2 = 0.5
    V_l  = env.Vdes - dV
    V_h = env.Vdes + dV
    N = 1
    P = matrix(np.diag([0.0, 1e24, 1e24]), tc='d')
    q = matrix(np.zeros(3))
    delta = env.T

    I = env.state[0]
    V = env.state[1]

    c_l =  V - eta_1 * (I - (env.G-G_diss) * V_l)
    c_h = -V + eta_2 * (I - (env.G+G_diss) * V_h)
    
    #print(c)

    G = -np.array([[env.Vs, 1, 0], [-env.Vs, 0, 1], [1, 0, 0], [-1, 0, 0]])
    G = matrix(G,tc='d')

    h = -np.array([c_l, c_h, 0, -1])
    #print(h)
    h = np.squeeze(h).astype(np.double)
    h = matrix(h,tc='d')

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)

    u_bar = sol['x']
    if np.abs(u_bar[1]) > 0.001:
        print("Violation of Safety: ")
        print(u_bar[1])
    return u_bar[0]


def cbf_3(env, dV = 5, G_diss = 0.02, eta_1= 0.5, eta_2=0.5):
    
    V_l = env.Vdes - dV
    V_h = env.Vdes + dV
    P = matrix(np.diag([0.0, 1e24, 1e24]), tc='d')
    q = matrix(np.zeros(3))
    delta_1 = eta_1*env.L/env.T
    delta_2 = eta_2*env.L/env.T

    I = env.state[0]
    V = env.state[1]

    c_l = -V - delta_1*env.G*V_l - (env.R - delta_1)*I
    c_h =  V + delta_2*env.G*V_h + (env.R - delta_2)*I
    
    #print(c)

    G = np.array([[-env.Vs, 1, 0], [env.Vs, 0, -1], [-1, 0, 0], [1, 0, 0]])
    G = matrix(G,tc='d')

    h = np.array([c_l, c_h, 0, 1])
    #print(h)
    h = np.squeeze(h).astype(np.double)
    h = matrix(h,tc='d')

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)

    u_bar = sol['x']
    if np.abs(u_bar[1]) > 0.001:
        print("Violation of Safety: ")
        print(u_bar[1])
    return u_bar[0]





def plot_signals(data, Ides, Vdes, dt = 1e-5, dv = 1):
    number_of_colors = data.shape[1]
    color = ['r', 'b']
    des = [Ides, Vdes]
    fig, ax = plt.subplots(nrows=1, ncols=data.shape[1], figsize = (8,4))
    time = np.array(range(data.shape[0]), dtype=np.float32)*dt
    for i in range(data.shape[1]):
        ax[i].plot(time, data[:, i],  c = color[i])
        ax[i].plot(time, np.full(data[:,i].shape[0], des[i] - dv), marker = '.')
        ax[i].plot(time, np.full(data[:,i].shape[0], des[i] + dv), marker = '.')
        ax[i].set_ylim(des[i]-10, des[i]+10)
    ax[0].set_title('Current', fontsize=20)
    ax[0].set_xlabel('Time', fontsize=20)
    ax[1].set_title('Voltage', fontsize=20)
    ax[1].set_xlabel('Time', fontsize=20)
    plt.show() 