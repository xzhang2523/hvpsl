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

colors_array = 'bgrcmyk' * 100
ref_point = np.array([1.0, 1.0])
hv_ind = Hypervolume(ref_point=ref_point)


def loss_subproblem(f, pref):
    return torch.max(f/pref)
    
    
    
    

    
    
def optimize_sub_problem(theta, problem_name, n_var=10):
    pref = Tensor(array([np.cos(theta), np.sin(theta)]))
    x = Variable(Tensor(np.random.random((1, n_var))), requires_grad=True)
    lr=1e-1
    
    optimizer = optim.SGD([x], lr=lr)
    sub_iteration_num = 1000
    for i in range(sub_iteration_num):
        f = loss_function(x, problem=problem_name).squeeze()
        loss = loss_subproblem(f, pref)
        optimizer.zero_grad()
        loss.backward()
        if torch.any(torch.isnan(x.grad)):
            break
        eps_x = 0.00
        optimizer.step()
        if problem_name == 'zdt1':
            x.data = torch.clamp(x.data, eps_x, 1-eps_x)
    res = f / pref
    return f, torch.min(res).detach().numpy()


def get_pref_from_theta(theta):
    return np.c_[np.cos(theta), np.sin(theta)]
    
    
def optimize_pf_hv(problem_name, sub_problem_numner):
    rho_array = [0] * sub_problem_numner
    f_array = [0] * sub_problem_numner
    theta_array = [0] * sub_problem_numner
    n_var = 30
    theta_eps = 5e-2
    for idx in tqdm(range(sub_problem_numner)):
        theta = np.random.random() * np.pi/2
        theta = np.clip(theta, theta_eps, np.pi/2-theta_eps)
        f, rho = optimize_sub_problem(theta, problem_name)
        f_array[idx] = f.detach().numpy()
        rho_array[idx] = rho
        theta_array[idx] = theta
    return array(f_array), array(rho_array), theta_array
        
def calculate_hv_from_rho(rho_array):
    area = 1-np.pi/4 * np.mean(rho_array**2) 
    return area
    
    
    
    
if __name__ == '__main__':
    problem_name = 'zdt1'
    problem_name = 'vlmop2'
    
    sub_problem_numner_array = [10, 50, 100, 150, 200,500]
    sub_problem_numner_array = [10, 50, 150, 200]
    # sub_problem_numner_array = [10, 20]
    # sub_problem_numner_array = [20,]
    # sub_problem_numner_array = [200,]
    sub_problem_numner_array = [50,]
    seed_num = 1
    # sub_problem_numner_array = [1,2,3,4]
    sub_problems = len(sub_problem_numner_array)
    
    hv_array_mean_array = [0] * sub_problems
    hv_array_std_array = [0] * sub_problems
    
    hv_array_mean_array_cartesian = [0] * sub_problems
    hv_array_std_array_cartesian = [0] * sub_problems
    
    
    
    
    plot_single = True
    for num_idx, sub_problem_numner in enumerate(sub_problem_numner_array):
        hv_array = [0] * seed_num
        hv_array_cartesian = [0] * seed_num
        
        for seed in range(seed_num):
            np.random.seed(seed=seed)
            pf_hat, rho_array, theta_array = optimize_pf_hv(problem_name, sub_problem_numner)
            hv_array[seed] = calculate_hv_from_rho(rho_array)
            hv_array_cartesian[seed] = hv_ind.do(pf_hat)
            
        hv_array_mean_array[num_idx] = np.mean(hv_array)
        hv_array_std_array[num_idx] = np.std(hv_array)
        hv_array_mean_array_cartesian[num_idx] = np.mean(hv_array_cartesian)
        hv_array_std_array_cartesian[num_idx] = np.std(hv_array_cartesian)
        pf_true = get_pf(problem_name, points_num=20)
        
        pref_array = get_pref_from_theta(theta_array)
        
        for pref in pref_array:
            plt.plot([0, pref[0]], [0, pref[1]])
        
        if plot_single:
            plt.scatter(pf_hat[:,0], pf_hat[:,1], label='Estimated')
            plt.scatter(pf_true[:,0], pf_true[:,1], label='True')
            plt.legend()
            plt.show()
            
    if not plot_single:
        hv_array_mean_array = array(hv_array_mean_array)
        hv_array_std_array = array(hv_array_std_array)
        hv_array_mean_array_cartesian = array(hv_array_mean_array_cartesian)
        hv_array_std_array_cartesian = array(hv_array_std_array_cartesian)
        
        plt.plot(sub_problem_numner_array, hv_array_mean_array, label='PSL Polar')
        plt.fill_between(sub_problem_numner_array, hv_array_mean_array-hv_array_std_array,  hv_array_mean_array+hv_array_std_array, alpha=0.2)
        plt.plot(sub_problem_numner_array, hv_array_mean_array_cartesian, label='PSL Cartesian')
        plt.fill_between(sub_problem_numner_array, hv_array_mean_array_cartesian-hv_array_std_array_cartesian,  hv_array_mean_array_cartesian+hv_array_std_array_cartesian, alpha=0.2)
        
        
        pf_true = get_pf(problem_name)
        hv_true = get_true_hv(problem_name)
        plt.plot(sub_problem_numner_array, np.ones_like(sub_problem_numner_array)*hv_true, label='True')
        plt.legend()
        plt.xlabel('Number of samples')
        plt.ylabel('HV')
        
        
        
        fig_prefix = 'C:\\Users\\xzhang2523\\Desktop\\IJCAI_submit\\HV_PSL\\Figures\\hv_approximation'
        fig_prefix = os.path.join(fig_prefix, problem_name)
        os.makedirs(fig_prefix, exist_ok=True)
        fig_name = os.path.join(fig_prefix, 'traj.pdf')
        plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
        print('fig saved in :{}'.format(fig_name) )
        
        
        
        
        
        
        plt.show()
    
    
        
            
        
    # pf_true = get_pf(problem_name)
    # hv_true = hv_ind.do(pf_true)
    
    
    # print('pf_true:{}'.format(hv_true))
    # print('pf_hat:{}'.format(calculate_hv_from_rho(rho_array)))
    
    # plt.scatter(pf_hat[:,0], pf_hat[:,1], label='estimated')
    # plt.scatter(pf_true[:,0], pf_true[:,1], label='True')
    # plt.legend()
    # plt.show()
    
    
    
        