from problem import loss_function, get_pf, get_true_hv
import numpy as np
from numpy import array
import torch
from torch import Tensor
from torch.autograd import Variable
from torch import optim
from pymoo.indicators.hv import Hypervolume
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import argparse




colors_array = 'bgrcmyk' * 100
ref_point = np.array([1.0, 1.0])
hv_ind = Hypervolume(ref_point=ref_point)



def loss_subproblem(f, pref, scalar='tche'):
    m = 1
    if scalar=='tche':
        # return 1/m * torch.max(f/pref)**m 
        return 1/m * torch.max(f/pref)**m 
    elif scalar=='ls':
        return f @ pref
    
    
    
def optimize_sub_problem(pref, problem_name, iteration_num = 5000, n_var=10, lr=1e-3):
    if problem_name == 'vlmop2':
        x = Variable( torch.rand((1, n_var)) * 2 / (n_var) - 1/ n_var, requires_grad=True) 
    else:
        x = Variable( torch.rand((1, n_var)), requires_grad=True) 
    
    optimizer = optim.SGD([x], lr=lr)
    for i in range(iteration_num):
        f = loss_function(x, problem=problem_name).squeeze()
        loss = loss_subproblem(f, pref)
        optimizer.zero_grad()
        loss.backward()
        if torch.any(torch.isnan(x.grad)):
            print('nan occur, plz check')
            assert False
        eps_x = 1e-5
        optimizer.step()
        if problem_name in ['zdt1', 'dtlz2']:
            x.data = torch.clamp(x.data, eps_x, 1-eps_x)
        elif problem_name in ['vlmop3']:
            x.data = torch.clamp(x.data, -3, 3)
        
    res = f / pref
    return f, torch.min(res).detach().numpy()


def get_pref_from_theta(theta):
    if len(theta.shape)==2:
        th1, th2 = theta[:,0], theta[:,1]
        pref1 = np.sin(th1) * np.sin(th2)
        pref2 = np.sin(th1) * np.cos(th2)
        pref3 = np.cos(th1)
        return np.c_[pref1, pref2, pref3]
    else:
        return np.c_[np.sin(theta), np.cos(theta)]
    



def optimize_pf_hv(problem_name, n_obj, args):
    n_problem = args.n_problem
    theta_eps = 1e-2
    if n_obj==2:
        theta_array = np.linspace(theta_eps, np.pi/2-theta_eps, n_problem)
        pref_array = Tensor(get_pref_from_theta(theta_array))
    else:
        theta_array = np.linspace(theta_eps, np.pi/2-theta_eps, n_problem)
        X, Y = np.meshgrid(theta_array, theta_array)
        theta_2d = []
        for i in range(n_problem):
            for j in range(n_problem):
                theta_2d.append(array([X[i][j], Y[i][j]]))
        theta_2d = array(theta_2d)
        pref_array = Tensor(get_pref_from_theta(theta_2d)) 
        
    
    n_problem = len(pref_array)
    rho_array = [0] * n_problem
    f_array = [0] * n_problem
    for idx in tqdm(range(n_problem)):
        f, rho = optimize_sub_problem(pref_array[idx], problem_name, iteration_num = args.iteration_num, lr=args.lr)
        f_array[idx] = f.detach().numpy()
        rho_array[idx] = rho
    return array(f_array), array(rho_array), theta_array
    
        
        
        
        
def calculate_hv_from_rho(rho_array):
    area = 1-np.pi/4 * np.mean(rho_array**2) 
    return area
    
    
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')
    
    parser.add_argument('--n-var', type=int, default=10)
    parser.add_argument('--iteration-num', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n-problem', type=int, default=3)
    parser.add_argument('--problem-name', type=str, default='vlmop2')
    args = parser.parse_args()
    problem_name = args.problem_name
    
    print('problem:{}'.format(problem_name))
    if problem_name in ['dtlz2', 'vlmop3']:
        n_obj=3
    else:
        n_obj=2
    
    if n_obj==3:
        args.lr=1e-1
    else:
        args.lr=1e-2

    
    
    
    
    pf_hat, rho_array, theta_array = optimize_pf_hv(problem_name, n_obj, args=args)
    if n_obj == 2:
        pref_array = np.c_[np.sin(theta_array), np.cos(theta_array)]
        for pref in pref_array:
            plt.plot([0, pref[0]], [0, pref[1]], color='tomato')
            
            
            
        plt.scatter(pf_hat[:,0], pf_hat[:,1], label='HV-scalar.')
        pf_true = get_pf(problem_name)
        plt.plot(pf_true[:,0], pf_true[:,1], label='True')
        
        plt.axis('equal')
        plt.xlabel('$f_1(x)$', fontsize=16)
        plt.ylabel('$f_2(x)$', fontsize=16)
        fig_prefix = 'C:\\Users\\xzhang2523\\Desktop\\IJCAI_submit\\HV_PSL\\Figures\\exp1'
        fig_prefix = os.path.join(fig_prefix, problem_name)
        os.makedirs(fig_prefix, exist_ok=True)
        fig_name = os.path.join(fig_prefix, 'uniform.pdf')
        plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
        print('saved in {}'.format(fig_name))
        
        plt.show()
        plt.figure()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pf_hat[:,0], pf_hat[:,1], pf_hat[:,2], label='Estimated')
        
        
        u, v = np.mgrid[0:np.pi/2:50j, 0:np.pi/2:50j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        ax.plot_surface(x, y, z, color="g", alpha=0.2, label='True')
        
        ax.set_xlabel('$f_1(x)$', fontsize=16)
        ax.set_ylabel('$f_2(x)$', fontsize=16)
        ax.set_zlabel('$f_3(x)$', fontsize=16)
        
        fig_prefix = 'C:\\Users\\xzhang2523\\Desktop\\IJCAI_submit\\HV_PSL\\Figures\\exp1'
        fig_prefix = os.path.join(fig_prefix, problem_name)
        os.makedirs(fig_prefix, exist_ok=True)
        fig_name = os.path.join(fig_prefix, 'uniform.pdf')
        plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
        print('saved in {}'.format(fig_name))
        
        
        # plt.legend()
        plt.show()
        

        # print()
        
    
    
        
            
            
            