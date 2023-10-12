import numpy as np
from numpy import array

import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle

import torch
from torch import Tensor
from torch.autograd.functional import jacobian
from torch.autograd import Variable
from torch import optim
from torch import nn
import torch.nn.functional as F

from pymoo.indicators.hv import Hypervolume
ref_point = np.array([3.5, 3.5])
hv_indicator = Hypervolume(ref_point=ref_point)



from pf_util import load_real_pf, LQR
from problem import loss_function
from reproblem import RE22, RE21, RE24, load_re_pf_norm
from psl_model import PrefNet
from moo_data import sgd_lr_dict



class PrefNet(nn.Module):
    def __init__(self, problem_name='zdt1', problem=None):
        super().__init__()
        # if problem_name in ['zdt1', 'zdt2']:
            # n_var = 3
        # elif problem_name == 'lqr2':
            # n_var = 2
        n_var = problem.n_variables
            
            
        self.problem_name = problem_name
        hidden_size = 128
        self.fc1 = nn.Linear(2, hidden_size)  # 6*6 from image dimension
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_var)
        self.lb = problem.lbound
        self.ub = problem.ubound
        
        
        # print()


    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        # if self.problem_name in ['zdt1', 'zdt2']:
            # x = F.sigmoid(self.fc3(x))
        # elif self.problem_name == 'lqr2':
            # x = F.sigmoid(self.fc3(x))-1
        x = F.sigmoid(self.fc3(x)) * (self.ub - self.lb) + self.lb
        return x
        
        
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
    k1_1 = -1./(1 + torch.exp(r[0]+r[1]*t))
    k2_2 = -1./(1 + torch.exp(r[2]+r[3]*t))
    return torch.stack([k1_1, k2_2])




    
        
        
def get_pref_from_angle(angle):
    pref1 = torch.cos(angle).unsqueeze(0)
    pref2 = torch.sin(angle).unsqueeze(0)
    pref = torch.cat([pref1, pref2], dim=0)
    return pref.T
    # print()
        
def mtche_loss(J, pref, decompose = 'tche', ref=torch.Tensor([1.52,1.52])):
    loss_arr = [0] * len(J)
    
    for idx, (ji, prefi) in enumerate(zip(J, pref)):
        # loss_arr[idx] = torch.max(ji / prefi)
        if decompose == 'tche':
            loss_arr[idx] = torch.max((ji - ref) * prefi)
        elif decompose == 'mtche':
            loss_arr[idx] = torch.max((ji - ref) / prefi)
        elif decompose == 'hv':
            # loss_arr[idx] = torch.clip(torch.max((ji - ref)**2 / prefi**2), 2) 
            loss_arr[idx] = torch.max((ji - ref)**2 / prefi**2) 
        elif decompose == 'ls':
            loss_arr[idx] = (ji - ref) @ prefi
            
    return np.pi/4 * torch.mean(torch.stack(loss_arr))
    
    
    
if __name__ == '__main__':
    solutions = torch.rand((2,10)) * 200 + 150
    res = get_indicator(solutions)
    lqr = LQR(n_obj=2)
    batch_size = 128
    precent = 0.1
    batch_01_num = int(batch_size*precent)
    
    
    decompose = 'hv'
    problem_name = 'RE24'
    if decompose == 'hv':
        lr = 1e-4
    else:
        lr = 1e-2
        
        
    
    
    
    
    
    n_iter = 75
    loss_array = [0] * n_iter
    
    problem_dict = {
        'RE21' : RE21(),
        'RE24' : RE24(),
    }
    problem = problem_dict[problem_name]
    
    x = torch.ones((10,4)) * 2
    problem.evaluate(x)
    
    
    
    model = PrefNet(problem_name=problem_name, problem=problem)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    
    pref_eps = 1e-2
    pref_eps_max = 1-pref_eps
    for iter in tqdm(range(n_iter)):
        pref = torch.rand((batch_size,2))
        for i in range(len(pref)):
            pref[i,:] = pref[i,:] / torch.sum(pref[i,:])
        pref = torch.clamp(pref, pref_eps, pref_eps_max)
        X = model(pref)
        J = problem.evaluate(X)
        
        if problem_name in ['zdt1', 'zdt2', 'RE21','RE24']:
            ref_point = Tensor([0.0,0.0])
        elif problem_name == 'lqr2':
            ref_point = Tensor([1.51, 1.51])
        else:
            assert False, '{} for ref not set'.format(problem_name)
            
        loss = mtche_loss(J, pref, decompose=decompose, ref=ref_point)
        
        optimizer.zero_grad()
        loss.backward()
        loss_array[iter] = loss.detach().numpy()
        
        # if problem_name == 'lqr2':

        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=2.0, error_if_nonfinite=False)
        optimizer.step()
    
    
    
    plt.plot(loss_array)
    plt.xlabel('Iteration')
    plt.ylabel('mTche Loss')
    plt.show()
    
    
    
    
    
    plt.figure()
    eps = 0.0
    pref = torch.linspace(eps, 1-eps, 1000)
    pref = torch.cat([pref.unsqueeze(0), (1-pref).unsqueeze(0)], dim=0).T
    
    with torch.no_grad():
        X = model(pref)
        pref_np = pref.numpy()
        if problem_name in ['zdt1', 'zdt2', 'RE21']:
            pref_np = pref_np * 0.4
            
        # J = loss_function(theta, problem=problem_name).numpy()
        J = problem.evaluate(X)
        J_np = J.numpy()
        
        
        X_np = X.numpy()
        pickle_file_name = os.path.join('D:\\code\\Paper_IJCAI\\draw_script\\data', '{}_{}.pickle'.format(problem_name, n_iter))
        with open(pickle_file_name, 'wb') as f:
            pickle.dump((pref_np, X_np, J_np), f)
        print('saved in {}'.format(pickle_file_name))
        
        
        
        use_plt = False
        if use_plt:
            for prefi, ji in zip(pref_np, X_np):
                plt.plot([prefi[0], ji[0]], [prefi[1], ji[1]])
            plt.scatter(X_np[:,0], X_np[:,1], label='Estimated')
            # plt.scatter(pf_real[:,0], pf_real[:,1], label='True')
            plt.axis('equal')
            plt.legend()
            
            fig_prefix = os.path.join('C:\\Users\\xzhang2523\\Desktop\\IJCAI_submit\\HV_PSL\\Figures\\hv_approximation', problem_name) 
            os.makedirs(fig_prefix, exist_ok=True)
            fig_name = os.path.join(fig_prefix, 'psl_X_{}.svg'.format(decompose))
            plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
            print('fig saved in :{}'.format(fig_name) )
            plt.show()
            
            
            
            
            plt.figure()
            pf_real = load_re_pf_norm(name=problem_name) 
            for prefi, ji in zip(pref_np, J):
                plt.plot([prefi[0], ji[0]], [prefi[1], ji[1]])
            plt.scatter(J[:,0], J[:,1], label='Estimated')
            # plt.scatter(pf_real[:,0], pf_real[:,1], label='True')
            plt.axis('equal')
            plt.legend()
            
            fig_prefix = os.path.join('C:\\Users\\xzhang2523\\Desktop\\IJCAI_submit\\HV_PSL\\Figures\\hv_approximation', problem_name) 
            os.makedirs(fig_prefix, exist_ok=True)
            fig_name = os.path.join(fig_prefix, 'psl_{}.svg'.format(decompose))
            plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
            print('fig saved in :{}'.format(fig_name) )
            plt.show()
        
    
    
    
    
    # print()
    
    # plt.plot(theta_np[0,:], theta_np[1,:])
    # plt.show()
    
    # # plt.plot
    # with torch.no_grad():
    #     J = get_J(theta).numpy()
    # hv_val = hv_indicator.do(J)
    # hv_scalar = 1e4
    # print('PGMA hv:{:.2f}'.format(hv_val / hv_scalar))
    
    # pf_real = load_real_pf()
    # hv_true_val = hv_indicator.do(pf_real)
    # print('True hv:{:.2f}'.format(hv_true_val / hv_scalar))
    
    
    # plt.scatter(J[:,0], J[:,1], label='PGMA')
    # plt.scatter(pf_real[:,0], pf_real[:,1], label='True')
    # plt.legend()
    # plt.show()
    
    
    
        
        
        
        
        
        
    
        
    
    