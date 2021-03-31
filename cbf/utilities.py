import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import matplotlib.pyplot as plt
import matplotlib
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



def cbf_decentralized(I, V, Vl, Vh, Vs, delta_1, delta_2, Gl, Gh, T):
    P = matrix(np.diag([1.0, 1e24, 1e24]), tc='d')
    q = matrix(np.zeros(3))
    delta = T
    c_l =  -V - delta_1 * (I - (Gl) * Vl)
    c_h = V + delta_2 * (I - (Gh) * Vh)

    G = -np.array([[Vs, 1, 0], [-Vs, 0, 1], [1, 0, 0], [-1, 0, 0]])
    G = matrix(G,tc='d')

    h = -np.array([c_l, c_h, 0, -1])
    h = np.squeeze(h).astype(np.double)
    h = matrix(h,tc='d')

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)

    u_bar = sol['x']
    # if np.abs(u_bar[1]) > 0.001:
    #     print("Violation of Safety: ")
    #     print(u_bar[1])
    return u_bar[0]
def cbf_decentralized_1(env, node, Il, Ih, eta_1= 0.5, eta_2=0.5):
    #u is the optimization var
    P = matrix(np.diag([1.0, 1e24, 1e24]), tc='d')
    q = matrix(np.zeros(3))
    delta_1 = eta_1*env.L[node][node]/env.T
    delta_2 = eta_2*env.L[node][node]/env.T

    I = env.state[:4][node]
    V = env.state[4:][node]

    ud= env.udes[node]
    c_l = -V - delta_1*Il - (env.R[node][node] - delta_1)*I
    c_h =  V + delta_2*Ih + (env.R[node][node] - delta_2)*I
    
    #print(c)

    G = np.array([[-env.Vs[node], 1, 0], [env.Vs[node], 0, -1], [-1, 0, 0], [1, 0, 0]])
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


def cbf_decentralized_2(env, node, Il, Ih, eta_1= 0.5, eta_2=0.5):
    #u= du +ud
    #du is the opt var
    P = matrix(np.diag([1.0, 1e24, 1e24]), tc='d')
    q = matrix(np.zeros(3))
    delta_1 = eta_1*env.L[node][node]/env.T
    delta_2 = eta_2*env.L[node][node]/env.T

    I = env.state[:4][node]
    V = env.state[4:][node]

    ud = env.udes[node]

    c_l = -V - delta_1*Il - (env.R[node][node] - delta_1)*I + ud*env.Vs[node]
    c_h =  V + delta_2*Ih + (env.R[node][node] - delta_2)*I - ud*env.Vs[node]
    
    #print(c)

    G = np.array([[-env.Vs[node], 1, 0], [env.Vs[node], 0, -1], [-1, 0, 0], [1, 0, 0]])
    G = matrix(G,tc='d')

    h = np.array([c_l, c_h, ud, 1-ud])
    #print(h)
    h = np.squeeze(h).astype(np.double)
    h = matrix(h,tc='d')

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)

    u_bar = sol['x']
    if np.abs(u_bar[1]) > 0.001:
        print("Violation of Safety: ")
        print(u_bar[1])
    return ud + u_bar[0]


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


def plot_voltage(data, save_fig=False,savefig_filename=None, FontSize=20):
    
    N_steps = np.shape(data['s'])[0]
    I = np.array(data['s'])[:,:4]
    V = np.array(data['s'])[:,4:]
    u = np.array(data['a'])

    I_des = np.array(data['s_des'])[0]
    V_des = np.array(data['s_des'])[1]
    u_des = np.array(data['a_des'])

    Il = np.array(data['Il'])[0]
    Ih = np.array(data['Ih'])[0]

    Vl = 228
    Vh = 232

    c_label = ['$I_1$', '$I_2$', '$I_3$', '$I_4$']
    cl_label = ['$I_1^l$', '$I_2^l$', '$I_3^l$', '$I_4^l$']
    ch_label = ['$I_1^h$', '$I_2^h$', '$I_3^h$', '$I_4^h$']

    v_label = ['$V_1$', '$V_2$', '$V_3$', '$V_4$']
    vl_label = ['$V_{l,1}$', '$V_{l,2}$', '$V_{l,3}$', '$V_{l,4}$']
    vh_label = ['$V_{h,1}$', '$V_{h,2}$', '$V_{h,4}$', '$V_{h,4}$']
    color = ['r', 'b']
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize = (6,10))
    time = np.array(range(N_steps), dtype=np.float32)*data['dt']
    time =  time.reshape(-1, 1)
    for i in range(4):
        index = i
        ax[i].plot(time, V[:, index],  c = 'k', label=v_label[index], linewidth=2.0)
        ax[i].plot(time, np.full(N_steps, Vl), '--', alpha=0.75, label=vl_label[index], linewidth=2.0)
        ax[i].plot(time, np.full(N_steps, Vh), '--', alpha=0.75, label=vh_label[index], linewidth=2.0)
        ax[i].set_ylim(Vl-2, Vh+2)
        ax[i].set_xlim(0, 0.5)
        ax[i].set_xlabel('Time (sec)', fontsize=FontSize)
        ax[i].set_ylabel('Voltage (V)', fontsize=FontSize)
        ax[i].set_label('Label via method')
        ax[i].legend(loc="upper right", fontsize=FontSize, ncol=3)
        ax[i].tick_params(axis='both', which='major', labelsize=FontSize)
        ax[i].tick_params(axis='both', which='minor', labelsize=FontSize)
        ax[i].axvline(x=0.2, color='m', linestyle='--', alpha = 0.57 , linewidth=2.0)
    if save_fig:
        assert isinstance(savefig_filename, str), \
                "filename for saving the figure must be a string"
        plt.savefig(savefig_filename, format = 'pdf')
    else:
        plt.show()


def plot_current(data, save_fig=False,savefig_filename=None,FontSize=12):
    print("IAMHERE")
    N_steps = np.shape(data['s'])[0]
    I = np.array(data['s'])[:,:4]
    V = np.array(data['s'])[:,4:]
    u = np.array(data['a'])

    I_des = np.array(data['s_des'])[0]
    V_des = np.array(data['s_des'])[1]
    u_des = np.array(data['a_des'])

    Il = np.array(data['Il'])[0]
    Ih = np.array(data['Ih'])[0]
    c_label = ['$I_1$', '$I_2$', '$I_3$', '$I_4$']
    cl_label = ['$I_1^l$', '$I_2^l$', '$I_3^l$', '$I_4^l$']
    ch_label = ['$I_1^h$', '$I_2^h$', '$I_3^h$', '$I_4^h$']
    cl_label = ['$I_{l,1}$', '$I_{l,2}$', '$I_{l,3}$', '$I_{l,4}$']
    ch_label = ['$I_{h,1}$', '$I_{h,2}$', '$I_{h,4}$', '$I_{h,4}$']
    color = ['r', 'b']
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize = (8,8))
    time = np.array(range(N_steps), dtype=np.float32)*data['dt']
    time =  time.reshape(-1, 1)
    for i in range(2):
        for j in range(2):
            index = 2*i + j
            ax[i,j].plot(time, I[:, index],  c = 'k', label=c_label[index], linewidth=2.0)
            ax[i,j].plot(time, np.full(N_steps, Il[index]), '--', alpha=0.75, label=cl_label[index], linewidth=2.0)
            ax[i,j].plot(time, np.full(N_steps, Ih[index]), '--', alpha=0.75, label=ch_label[index], linewidth=2.0)
            ax[i,j].set_ylim(Il[index]-1, Ih[index]+1)
            ax[i,j].set_xlim(0, 0.5)
            ax[i,j].set_xlabel('Time (sec)', fontsize=FontSize)
            ax[i,j].set_ylabel('Current (A)', fontsize=FontSize)
            ax[i,j].set_label('Label via method')
            ax[i,j].legend(loc="lower right", fontsize=FontSize, ncol=1)
            ax[i,j].tick_params(axis='both', which='major', labelsize=FontSize)
            ax[i,j].tick_params(axis='both', which='minor', labelsize=FontSize)
            ax[i,j].axvline(x=0.2, color='m', linestyle='--', alpha = 0.57 , linewidth=2.0)
    if save_fig:
        assert isinstance(savefig_filename, str), \
                "filename for saving the figure must be a string"
        plt.savefig(savefig_filename, format = 'pdf')
    else:
        plt.show()

def plot_dutyratio(data, save_fig=False,savefig_filename='dutyratio.pdf', FontSize=12):
    print("HERE" + "**"*40)
    N_steps = np.shape(data['s'])[0]
    I = np.array(data['s'])[:,:4]
    V = np.array(data['s'])[:,4:]
    u = np.array(data['a'])

    I_des = np.array(data['s_des'])[0]
    V_des = np.array(data['s_des'])[1]
    u_des = np.array(data['a_des'])

    Il = np.array(data['Il'])[0]
    Ih = np.array(data['Ih'])[0]
    N_steps = np.shape(data['s'])[0]
    time = np.array(range(N_steps), dtype=np.float32)*data['dt']
    time = time.reshape(-1,1)
    label = ['$u_1$', '$u_2$',' $u_3$', '$u_4$']
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize = (8,4.5))
    for i in range(4): 
        ax.plot(time, u[:,i],  linewidth=2.0,Label=label[i])
    # ax.set_ylim(0.54, 0.6)
    ax.set_xlabel('Time (sec)', fontsize=FontSize)
    ax.set_ylabel('Duty-ratio', fontsize=FontSize)
    ax.set_label('Label via method')
    ax.set_ylim(0.525, 0.6)
    ax.set_xlim(0, 0.5)
    ax.legend(loc="lower right", fontsize=FontSize, ncol=4)
    ax.tick_params(axis='both', which='major', labelsize=FontSize)
    ax.tick_params(axis='both', which='minor', labelsize=FontSize)
    ax.axvline(x=0.2, color='m', linestyle='--', alpha = 0.57 , linewidth=2.0)
    if save_fig:
        assert isinstance(savefig_filename, str), \
                "filename for saving the figure must be a string"
        plt.savefig(savefig_filename, format = 'pdf')
    else:
        plt.show()
    # plt.savefig('dutyratio.pdf', format = 'pdf')
    plt.show()