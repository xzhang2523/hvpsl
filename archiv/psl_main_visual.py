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
from pf_util import load_real_pf, LQR, compute_sparsity
from reproblem import RE21, RE24, load_re_pf_norm, RE37, RE34, RE33
from pymoo.indicators.hv import Hypervolume

# from pf_util import 
from torch import optim
import os
from torch import nn
import torch.nn.functional as F
from problem import loss_function, uniform_sphere_pref
from solvers import EPOSolver
import csv
import argparse
import warnings
import time
# import the_module_that_warns
warnings.simplefilter("ignore", UserWarning)

import pickle





class PrefNet(nn.Module):
    def __init__(self, args, problem=None):
        super().__init__()
        self.problem_name = args.problem_name
        if self.problem_name.startswith('RE'):
            self.lb = problem.lbound
            self.ub = problem.ubound
            n_var = problem.n_variables
            hidden_size = 64
        else:
            n_var = args.n_var
            hidden_size = 64
        self.fc1 = nn.Linear(args.n_obj, hidden_size)  # 6*6 from image dimension
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, n_var)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        if self.problem_name in ['zdt1', 'zdt2', 'dtlz2']:
            x = F.sigmoid(self.fc4(x))
        elif self.problem_name in ['lqr2', 'lqr3']:
            x = F.sigmoid(self.fc4(x))-1
        elif self.problem_name.startswith('RE'):
            x = F.sigmoid(self.fc4(x)) * (self.ub - self.lb) + self.lb
        else:
            assert False
            
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


def mtche_loss(J, pref, args, x=None, theta=None):
    m = args.n_obj
    loss_arr = [0] * len(J)
    decompose = args.decompose
    for idx, (ji, prefi) in enumerate(zip(J, pref)):
        if decompose == 'tche':
            loss_arr[idx] = torch.max((ji - args.ideal_point) * prefi)
        elif decompose in ['mtche', 'mtchenograd']:
            loss_arr[idx] = torch.max((ji - args.ideal_point) / prefi)
        elif decompose == 'hv1':
            arg_idx = torch.argmax((ji - args.nadir_point)/prefi)
            rho = ((ji - args.nadir_point)/prefi)[arg_idx]
            if m==2:
                if rho > 0:
                    loss_arr[idx] = rho**m
                else:
                    loss_arr[idx] = -rho**m
            else:
                loss_arr[idx] = rho**m
        elif decompose in ['hv2', 'hv2nograd']:
            if m ==2:
                loss_arr[idx] = torch.max((ji - args.ideal_point)**2 / prefi**2)
            elif m==3:
                loss_arr[idx] = torch.max((ji - args.ideal_point)**m / prefi**m) * torch.sin(theta[idx, 0])
        elif decompose == 'ls':
            loss_arr[idx] = (ji - args.ideal_point) @ prefi
        elif decompose == 'epo':
            solver = EPOSolver(n_tasks=args.n_obj, n_params = args.n_var)
            loss_arr[idx] = solver.get_weighted_loss((ji - args.ideal_point), ray=1/prefi, parameters=x)
    return torch.mean(torch.stack(loss_arr))
    
    
def generate_rand_pref(n_obj):
    if n_obj == 2:
        theta = torch.rand(batch_size) * np.pi/2
        pref = torch.cat([torch.sin(theta).unsqueeze(1), torch.cos(theta).unsqueeze(1)], axis=1)
    else:
        theta = torch.rand((batch_size,2)) * np.pi/2
        th1 = theta[:,0]
        th2 = theta[:,1]
        p1 = torch.sin(th1) * torch.sin(th2)
        p2 = torch.sin(th1) * torch.cos(th2)
        p3 = torch.cos(th1)
        pref = torch.cat([p1.unsqueeze(0),p2.unsqueeze(0),p3.unsqueeze(0)], axis=0).T
        # print()
    return theta, pref
    
def objective(x, problem_name):
    if args.problem_name.startswith('RE'):
        J = problem.evaluate(x)
    else:
        J = loss_function(x, problem=problem_name, )
    return J
    
def get_theta_2(J):
    x = J[:,0]
    y = J[:,1]
    z = J[:,2]
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arctan(np.sqrt(x**2+y**2)/ np.clip(z**2, np.ones_like(x)*1e-3, a_max=None) )
    fie = np.arctan(y/ np.clip(x, np.ones_like(x)*1e-3, a_max=None) )
    
    
    theta_range = np.max(theta) - np.min(theta)
    fie_range = np.max(fie) - np.min(fie)
    return min(theta_range, fie_range)
     

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')
    
    parser.add_argument('--n-var', type=int, default=5)
    parser.add_argument('--n-obj', type=int, default=2)
    parser.add_argument('--n-iter', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--problem-name', type=str, default='dtlz2')
    parser.add_argument('--decompose', type=str, default='mtchenograd')
    parser.add_argument('--no-plot', action='store_false')
    args = parser.parse_args()
    problem_name = args.problem_name
    if args.problem_name == 'lqr2':
        args.n_var = 2
    elif args.problem_name == 'lqr3':
        args.n_var = 3
    if args.problem_name in ['dtlz2', 'lqr3'] or args.problem_name.startswith('RE3'):
        args.n_obj = 3
    hv_indicator = Hypervolume(ref_point=np.ones(args.n_obj)*3.5)
    
    if args.n_obj==2:
        lr=1e-1
    else:
        if args.decompose in ['hv2', 'hv2nograd']:
            if problem_name.startswith('RE3'):
                lr=1e-1
            else:
                lr=1e-2
        if args.decompose.startswith('mtche'):
            if problem_name.startswith('RE3'):
                lr=1e-1
            else:
                lr=1e-1
        elif args.decompose in ['hv1']:
            if problem_name.startswith('RE3'):
                lr=0.1
            else:
                lr=0.01
        elif args.decompose in ['tche']:
            if problem_name.startswith('RE3'):
                lr=1e-3
            else:
                lr=1e-1
        else:
            if problem_name.startswith('RE3'):
                lr=1e-2
            else:
                lr=1e-2
                
    # lr = args.lr
    n_iter = args.n_iter
    if problem_name in ['zdt1', 'zdt2'] or problem_name.startswith('RE2'):
        ideal_point = Tensor([0.0, 0.0])
    elif problem_name == 'lqr2':
        ideal_point = Tensor([1.51, 1.51])
    elif problem_name == 'lqr3':
        ideal_point = Tensor([1.9, 1.9, 1.9])
    elif problem_name == 'dtlz2' or problem_name.startswith('RE3'):
        ideal_point = Tensor([0.0,0.0,0.0])
    else:
        assert False, '{} for ref not set'.format(problem_name)
        
    if problem_name in ['zdt1', 'zdt2'] or problem_name.startswith('RE2'):
        nadir_point = Tensor([1.3, 1.3])
    elif problem_name == 'lqr2':
        nadir_point = Tensor([4.0, 4.0])
    elif problem_name == 'lqr3':
        nadir_point = Tensor([4.0, 4.0, 4.0])
    elif problem_name == 'dtlz2':
        nadir_point = Tensor([1.0, 1.0, 1.0])
    elif problem_name.startswith('RE3'):
        nadir_point = Tensor([1.0, 1.0, 1.0])
    else:
        assert False, '{} for ref not set'.format(problem_name)
        
    args.ideal_point = ideal_point
    args.nadir_point = nadir_point
    
    batch_size = 128
    precent = 0.0
    batch_01_num = int(batch_size*precent)
    
    pref_eps = 1e-3
    if args.decompose == 'hv2':
        pref_eps = 1e-2
    
    pref_eps_max = 1-pref_eps
    loss_array = [0] * n_iter
    if args.problem_name.startswith('RE'):
        problem_dict = {
            'RE21' : RE21(),
            'RE24' : RE24(),
            'RE37' : RE37(),
            'RE34' : RE34(),
            'RE33' : RE33(),
        }
        problem = problem_dict[args.problem_name]
    else:
        problem = None
    model = PrefNet( args=args, problem=problem)
    optimizer_name = 'Adam'
    if optimizer_name=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
    
    ts = time.time()
    
    
    for iter in tqdm(range(n_iter)):
        theta, pref = generate_rand_pref(args.n_obj)
        pref = torch.clamp(pref, pref_eps, pref_eps_max)
        x = model(pref)
        J = objective(x, args.problem_name)
        loss = mtche_loss(J, pref, args=args, x=x, theta=theta)
        optimizer.zero_grad()
        loss.backward()
        loss_array[iter] = loss.detach().numpy()
        if not args.decompose.endswith('nograd'):
            max_norm = 5.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=2.0, error_if_nonfinite=False)
            
        
        optimizer.step()
    elaps = np.round(time.time() - ts, 2)
    plt.plot(loss_array)
    plt.xlabel('Iteration')
    plt.ylabel('Average loss')
    
    
    folder_prefix = 'C:\\Users\\xzhang2523\\Desktop\\IJCAI_submit\\HV_PSL\\Figures\\exp2'
    folder_prefix = os.path.join(folder_prefix, optimizer_name, problem_name)
    os.makedirs(folder_prefix, exist_ok=True)
    fig_name = os.path.join(folder_prefix, 'process_{}_{}.pdf'.format('psl', args.decompose))
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
    print('saved in {}'.format(fig_name))
    
    
    # save the pickle file
    
    plt.figure()
    if args.n_obj == 2:
        pref = Tensor(uniform_sphere_pref(m=args.n_obj, n=60))
    else:
        pref = Tensor(uniform_sphere_pref(m=args.n_obj, n=80))
    
    
    with torch.no_grad():
        if args.n_obj == 3:
            x = model(pref)
            J = objective(x, args.problem_name).numpy()
            
            pickle_name = os.path.join(folder_prefix, 'mid_{}.pickle'.format( n_iter ))
            with open(pickle_name, 'wb') as f:
                pickle.dump((x.numpy(), J), f)
                
            print('pickle saved in {}'.format(pickle_name))
            
            
            ax = plt.axes(projection='3d')
            ax.scatter3D(J[:,0], J[:,1], J[:,2], color = "green", label='PSL')
            ax.set_xlabel('$f_1(x)$', fontsize=16)
            ax.set_ylabel('$f_2(x)$', fontsize=16)
            ax.set_zlabel('$f_3(x)$', fontsize=16)
            fig_name = os.path.join(folder_prefix, '{}_{}.pdf'.format('psl', args.decompose))
            plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
            print('saved in {}'.format(fig_name))
        elif args.n_obj == 2:
            x = model(pref)
            pref_np = pref.numpy()
            if problem_name in ['zdt1', 'zdt2','vlmop1','vlmop2'] or problem_name.startswith('RE'):
                pref_np = pref_np * 0.4
            if problem_name.startswith('RE'):
                J = problem.evaluate(x).numpy()
            else:
                J = loss_function(x, problem=problem_name).numpy()
                                    
            if args.problem_name.startswith('RE'):
                pf_real = load_re_pf_norm(problem_name=problem_name) 
            else:
                pf_real = load_real_pf(problem_name=problem_name)
        
            for prefi, ji in zip(pref_np, J):
                if args.decompose == 'hv1':
                    nadir_np = args.nadir_point.numpy()
                    plt.plot([nadir_np[0]-prefi[0], ji[0]], [nadir_np[1]-prefi[1], ji[1]], color='tomato')
                else:
                    plt.plot([prefi[0], ji[0]], [prefi[1], ji[1]], color='tomato')
                    
            plt.scatter(J[:,0], J[:,1], label='PSL', color='gold')
            plt.plot(pf_real[:,0], pf_real[:,1], label='True', color='tab:blue')
            if args.decompose == 'hv1':
                plt.plot(nadir_np[0] - pref_np[:,0], nadir_np[1] - pref_np[:,1], color='skyblue', label='Pref.')
            else:
                plt.plot(pref_np[:,0], pref_np[:,1], color='skyblue', label='Pref.')
                
            plt.axis('equal')
            plt.legend()
            plt.xlabel('$f_1(x)$', fontsize=16)
            plt.ylabel('$f_2(x)$', fontsize=16)
            fig_name = os.path.join(folder_prefix, '{}_{}.pdf'.format('psl', args.decompose))
            plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
            print('fig saved in :{}'.format(fig_name) )
        
        
        # statistics
        if args.n_obj == 2:
            J_theta = array([np.pi/2 - np.arctan2(*elem) for elem in J])
            hv_range = np.round(np.max(J_theta) - np.min(J_theta),2)
        else:
            hv_range = np.round(get_theta_2(J),2)
        hv_val = np.round(hv_indicator.do(J), 2) 
        if args.n_obj == 2:
            sparsity = compute_sparsity(J) * 1e4
        else:
            sparsity = compute_sparsity(J) * 1e7
            
        print('PSL True hv:{:.2f}'.format(hv_val))
        csv_file_name= os.path.join(folder_prefix, '{}.csv'.format(args.decompose))
        with open(csv_file_name, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(['HV', 'Range', 'Sparsity','Time'])
            spamwriter.writerow([str(hv_val), str(hv_range), str(np.round(sparsity,3)),elaps])
        print('csv saved in {}'.format(csv_file_name))