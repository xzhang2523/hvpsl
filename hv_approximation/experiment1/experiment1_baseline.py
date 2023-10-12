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
from mo_optimizers.hv_maximization import HvMaximization, HvMaximizationRaw
from problem import loss_function
import argparse

from polar_util import get_sphere_surface


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')
    
    parser.add_argument('--n-var', type=int, default=10)
    parser.add_argument('--iteration-num', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n-problem', type=int, default=100)
    parser.add_argument('--problem-name', type=str, default='dtlz2')
    args = parser.parse_args()
    problem_name = args.problem_name
    
    n_mo_sol = args.n_problem
    n_var = args.n_var
    if problem_name == 'dtlz2':
        n_obj=3
    else:
        n_obj=2
    n_iter = args.iteration_num
        
    
    ref_point = np.ones(n_obj) * 2
    hv_indicator = Hypervolume(ref_point=ref_point)
    hv_indicator_star = Hypervolume(ref_point=np.ones(n_obj))
    
    
    # mtd = 'HV-Grad-Raw'
    mtd = 'HV-Grad'
    if mtd == 'HV-Grad':
        model = HvMaximization(n_mo_sol, n_obj, ref_point=ref_point)
    elif mtd == 'HV-Grad-Raw':
        model = HvMaximizationRaw(n_mo_sol, n_obj, ref_point=ref_point)
        
    
    
    if problem_name == 'vlmop2':
        x = Variable( torch.rand((n_mo_sol, n_var)) * 2 / (n_var) - 1/ n_var, requires_grad=True)
    else:
        x = Variable( torch.rand((n_mo_sol, n_var)), requires_grad=True) 
    
    lr=1e-2
    optimizer = optim.SGD([x], lr=lr)
    
    
    
    
    traj_array = [0] * n_iter
    hv_val_array = [0] * n_iter
    for idx in tqdm(range(n_iter)):
        loss = loss_function(x, problem=problem_name)
        loss_numpy = torch.clone(loss).detach().numpy()
        hv_val = hv_indicator.do(loss_numpy.T)
        hv_val_array[idx] = hv_val
        traj_array[idx] = loss_numpy
        weights = model.compute_weights( loss_numpy )
        hv_loss = torch.sum(weights * loss)
        optimizer.zero_grad()
        hv_loss.backward()
        optimizer.step()
        eps = 1e-3
        if problem_name in ['zdt1', 'dtlz2']:
            x.data = torch.clamp(x, eps, 1-eps)
        

    if n_obj == 2:
        traj = traj_array[-1]
        hv_star = hv_indicator_star.do(traj.T)
        print( 'hv_star{:.2f}'.format(hv_star) )
        plt.scatter(traj[0,:], traj[1,:])
        
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel('$f_1(x)$')
        plt.ylabel('$f_2(x)$')
        plt.axis('equal')

        fig_prefix = 'C:\\Users\\xzhang2523\\Desktop\\IJCAI_submit\\HV_PSL\\Figures\\hv_approximation'
        fig_prefix = os.path.join(fig_prefix, problem_name)
        os.makedirs(fig_prefix, exist_ok=True)
        fig_name = os.path.join(fig_prefix, '{}_2d.pdf'.format(mtd) )
        plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
        print('fig saved in :{}'.format(fig_name) )
        plt.show()
    else:
        pf_hat = traj_array[-1].T
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pf_hat[:,0], pf_hat[:,1], pf_hat[:,2], label='Estimated')
        x,y,z = get_sphere_surface()
        ax.plot_surface(x, y, z, color="g", alpha=0.2, label='True')
        
        ax.set_xlabel('$f_1(x)$', fontsize=16)
        ax.set_ylabel('$f_2(x)$', fontsize=16)
        ax.set_zlabel('$f_3(x)$', fontsize=16)
        
        fig_prefix = 'C:\\Users\\xzhang2523\\Desktop\\IJCAI_submit\\HV_PSL\\Figures\\exp1'
        fig_prefix = os.path.join(fig_prefix, problem_name)
        os.makedirs(fig_prefix, exist_ok=True)
        fig_name = os.path.join(fig_prefix, '{}_3d.pdf')
        plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
        print('saved in {}'.format(fig_name))
        
        
        
        
        
        
        plt.show()
        print()