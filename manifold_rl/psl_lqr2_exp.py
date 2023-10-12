import numpy as np
from numpy import array
from matplotlib import pyplot as plt
import torch
from torch import Tensor
from torch.autograd.functional import jacobian
import scipy
import scipy.io
from torch.autograd import Variable
from tqdm import tqdm
from pf_util import load_real_pf, LQR
from pymoo.indicators.hv import Hypervolume
ref_point = np.array([3.5, 3.5])
hv_indicator = Hypervolume(ref_point=ref_point)
# from pf_util import 
from torch import optim
import os
from torch import nn
import torch.nn.functional as F


        
        
        
    
    



def element_wise_division(J, Aup=array([1,2])):
    length = J.shape[-1]
    J_new = [0] * length
    for i in range(length):
        res = J[:,i] / Aup
        J_new[i] = res
    return torch.stack(J_new) 
        
        
        
def get_indicator(J, beta=1.0):
    AUp = Tensor(array([350, 350])) 
    Up = Tensor(array([350, 350])) 
    I_antiutopia = torch.sum((element_wise_division(J, AUp) - torch.ones(2))**2)
    I_utopia = torch.sum( (element_wise_division(J, Up) - torch.ones(2))**2 )
    I = I_antiutopia * (1 - beta * I_utopia)
    return I
    
    
    
    
def get_theta(r, t):
    # t = torch.linspace(0,1,100)
    k1_1 = -1./(1 + torch.exp(r[0]+r[1]*t))
    k2_2 = -1./(1 + torch.exp(r[2]+r[3]*t))
    return torch.stack([k1_1, k2_2])





def get_J(theta):
    n_subproblems = theta.shape[-1]
    J_array = [0] * n_subproblems
    for idx in range(n_subproblems):
        K = torch.diag(theta[:, idx])
        J = lqr.getJ(K)
        J_array[idx] = J.unsqueeze(0)
    J_array = torch.cat(J_array, dim=0)
    return J_array
    
        
        
def get_pref_from_angle(angle):
    pref1 = torch.cos(angle).unsqueeze(0)
    pref2 = torch.sin(angle).unsqueeze(0)
    pref = torch.cat([pref1, pref2], dim=0)
    return pref.T
    # print()
        
        
def mtche_loss(J, pref , decompose='tche', ref=torch.Tensor([1.51, 1.51])):
    loss_arr = [0] * len(J)
    for idx, (ji, prefi) in enumerate(zip(J, pref)):
        # loss_arr[idx] = torch.max((ji - ref) * prefi)
        # loss_arr[idx] = torch.max((ji - ref)**2 / prefi**2)
        if decompose == 'tche':
            loss_arr[idx] = torch.max((ji - ref) * prefi)
        elif decompose == 'mtche':
            loss_arr[idx] = torch.max((ji - ref) / prefi)
            
    return np.pi/4 * torch.mean(torch.stack(loss_arr))
    
    
    
    
    
    
if __name__ == '__main__':
    problem_name = 'lqr2'
    
    lqr = LQR(n_obj=2)
    batch_size = 32
    percent = 0.95
    batch_r = int(batch_size * percent / 2)
    rho = Variable(Tensor([1, 2, 0, 3]), requires_grad=True)
    mtd = 'exp'
    # decompose = 'mtche'
    decompose = 'mtche'
    
    
    lr = 0.05
    angle_eps = 0.01
    angle_max = np.pi/2 - angle_eps
    
    
    n_iter = 500
    optimizer = optim.SGD([rho], lr=lr)
    loss_arr = [0] * n_iter
    for iter in tqdm(range(n_iter)):
        angle = torch.rand(batch_size) * (angle_max - angle_eps) + angle_eps
        
        if decompose == 'tche':
            angle[:batch_r] = angle_eps
            angle[batch_r:2*batch_r] = np.pi/2 -  angle_eps
        elif decompose == 'mtche':
            batch_r = 3
            angle[:batch_r] = angle_eps
            angle[batch_r:2*batch_r] = np.pi/2 -  angle_eps    
        
        pref = get_pref_from_angle(angle)
        J = get_J(get_theta(rho, angle/(np.pi/2)))
        loss = mtche_loss(J, pref, decompose=decompose)
        loss_arr[iter] = float(loss.detach().numpy())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    
    
    
    
    
    
    plt.plot(loss_arr)
    plt.show()
    plt.figure()
    
    
    print( 'rho:{}'.format( np.round(rho.detach().numpy(), 2) ) )
    print('optimal rho:{}'.format([1.14, -3.02, -1.77, 2.91]))
    angle = torch.linspace(0, np.pi/2, 100)
    
    pref_array = get_pref_from_angle(angle).numpy()
    theta = get_theta(rho, angle / (np.pi/2))
    theta_np = theta.detach().numpy().T
    J_array= [0] * len(theta_np)
    for idx, theta_i in enumerate(theta_np):
        J = lqr.getJ(torch.Tensor.diag(Tensor(theta_i)))
        J_array[idx] = J
    J_array = torch.stack(J_array).numpy()
    for pref, J in zip(pref_array, J_array):
        plt.plot([pref[0],J[0]], [pref[1],J[1]], color='tomato')
    
    
    
    hv_scalar = 1
    hv_val = hv_indicator.do(J_array)
    print('PSL True hv:{:.2f}'.format(hv_val / hv_scalar))
    pf_real = load_real_pf(problem_name=problem_name)
    hv_true_val = hv_indicator.do(pf_real)
    print('True hv:{:.2f}'.format(hv_true_val / hv_scalar))
    
    
    plt.scatter(J_array[:,0], J_array[:,1], label='Estimated')
    plt.scatter(pf_real[:,0], pf_real[:,1], label='True')
    plt.axis('equal')
    plt.legend()
    plt.xlabel('$f_1(x)$')
    plt.ylabel('$f_2(x)$')
    
    
    
    problem_name = 'lqr2'
    fig_prefix = 'C:\\Users\\xzhang2523\\Desktop\\IJCAI_submit\\HV_PSL\\Figures\\hv_approximation'
    fig_prefix = os.path.join(fig_prefix, problem_name)
    os.makedirs(fig_prefix, exist_ok=True)
    
    
    fig_name = os.path.join(fig_prefix, 'psl_{}_{}.pdf'.format(mtd, decompose))
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
    print('fig saved in :{}'.format(fig_name) )
    plt.show()
    
    
        
        
        
        
        
        
    
        
    
    