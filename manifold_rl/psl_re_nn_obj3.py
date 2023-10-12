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
# from pf_util import load_real_pf, LQR
from pymoo.indicators.hv import Hypervolume
ref_point = np.array([3.5, 3.5])
hv_indicator = Hypervolume(ref_point=ref_point)
# from pf_util import 
from torch import optim
import os
from torch import nn
import torch.nn.functional as F

from problem import loss_function, get_pf
from reproblem import RE22, RE21, RE24, RE33, RE31, RE37, load_re_pf_norm, load_re_pf

from pymoo.util.ref_dirs import get_reference_directions









class PrefNet(nn.Module):
    def __init__(self, problem_name='zdt1', problem=None):
        super().__init__()
        # if problem_name in ['zdt1', 'zdt2']:
            # n_var = 3
        # elif problem_name == 'lqr2':
            # n_var = 2
        
        
        if problem_name.startswith('RE'):
            n_var = problem.n_variables
            self.lb = problem.lbound
            self.ub = problem.ubound
        else:
            n_var = 5
        
        if problem_name in ['dtlz2', 'RE33', 'RE31','RE37','maf1']:
            n_obj =3
        else:
            assert False
            
        
        
        self.problem_name = problem_name
        hidden_size = 128
        self.fc1 = nn.Linear(n_obj, hidden_size)  # 6*6 from image dimension
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, n_var)
        


    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        if self.problem_name in ['zdt1', 'zdt2','dtlz2']:
            x = F.sigmoid(self.fc4(x))
        elif self.problem_name == 'lqr2':
            x = F.sigmoid(self.fc4(x))-1
        elif self.problem_name.startswith('RE'):
            x = F.sigmoid(self.fc4(x)) * (self.ub - self.lb) + self.lb
        return x
        
        
        
        
def element_wise_division(J, Aup=array([1,2])):
    length = J.shape[-1]
    J_new = [0] * length
    for i in range(length):
        res = J[:,i] / Aup
        J_new[i] = res
    return torch.stack(J_new) 
        
        

        
        

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
            loss_arr[idx] = torch.max((ji - ref)**3 / prefi**3) 
        elif decompose == 'ls':
            loss_arr[idx] = (ji - ref) @ prefi
            
    return  torch.mean(torch.stack(loss_arr))
    
    
    
if __name__ == '__main__':
    batch_size = 128
    precent = 0.1
    batch_01_num = int(batch_size*precent)
    
    
    decompose = 'tche'
    if decompose == 'hv':
        lr = 1e-2
    else:
        lr = 1e-2
        
    
    problem_name = 'RE37'
    
    pref_eps = 0.01
    pref_eps_max = 1-pref_eps
    
    
    
    if problem_name.startswith('RE'):
        problem_dict = {
            'RE21' : RE21(),
            'RE24' : RE24(),
            'RE33' : RE33(),
            'RE31' : RE31(),
            'RE37' : RE37(),
        }
        problem = problem_dict[problem_name]
        model = PrefNet(problem_name=problem_name, problem=problem)
    else:
        model = PrefNet(problem_name=problem_name, problem=None)
        
    
    
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    n_obj = 3
    n_iter = 2000
    loss_array = [0] * n_iter
    
    for iter in tqdm(range(n_iter)):
        pref = torch.rand((batch_size, n_obj))
        for i in range(len(pref)):
            pref[i,:] = pref[i,:] / torch.sum(pref[i,:])
        pref = torch.clamp(pref, pref_eps, pref_eps_max)
        
        
        X = model(pref)
        if problem_name.startswith('RE'):
            J = problem.evaluate(X)
        else:
            J = loss_function(X, problem=problem_name)
        
        
        if problem_name in ['zdt1', 'zdt2', 'RE21','RE24']:
            ref_point = Tensor([0.0,0.0])
        elif problem_name == 'lqr2':
            ref_point = Tensor([1.51, 1.51])
        elif problem_name in ['dtlz2', 'RE33', 'RE31', 'RE37','maf1']:
            ref_point = Tensor([0, 0, 0])
        else:
            assert False, '{} for ref not set'.format(problem_name)
        loss = mtche_loss(J, pref, decompose=decompose, ref=ref_point)
        
        optimizer.zero_grad()
        loss.backward()
        loss_array[iter] = loss.detach().numpy()
        
        
        
        max_norm = 2.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=2.0, error_if_nonfinite=False)
        
        
        
        optimizer.step()
    
    

    plt.plot(loss_array)
    plt.show()
    plt.figure()
    
    
    
    
    
    pref_test = Tensor(get_reference_directions("energy", 3, 1000, seed=1))
    with torch.no_grad():
        if problem_name.startswith('RE'):
            solutions = problem.evaluate(model(pref_test)).numpy()
        else:
            solutions = loss_function(model(pref_test), problem=problem_name).numpy()
        
        
    ax = plt.axes(projection ="3d")
    ax.scatter3D(solutions[:,0], solutions[:,1], solutions[:,2], s=1, color = "orange", label='PSL')
    # surf = ax.plot_trisurf(solutions[:,0], solutions[:,1], solutions[:,2], color = "orange", label='PSL')
    # surf = ax.plot_trisurf(solutions[:,0], solutions[:,1], solutions[:,2], color = "orange")
    # s_xx, s_yy = np.meshgrid(solutions[:,0], solutions[:,1])
    # surf = ax.plot_surface(s_xx, s_yy, solutions[:,2], color = "orange", label='PSL')
    
    
    
    
    if problem_name.startswith('RE'):
        pf_norm = load_re_pf_norm(problem_name)
    else:
        pf_norm = get_pf(problem_name)
        
        
        
    ax.scatter3D(pf_norm[:,0], pf_norm[:,1], pf_norm[:,2], color='green', label='True')
    
    
    
    
    ax.set_xlabel('$f_1(x)$')
    ax.set_ylabel('$f_2(x)$')
    ax.set_zlabel('$f_3(x)$')

    plt.legend()

    fig_prefix = os.path.join('C:\\Users\\xzhang2523\\Desktop\\IJCAI_submit\\HV_PSL\\Figures\\hv_approximation', problem_name) 
    os.makedirs(fig_prefix, exist_ok=True)
    fig_name = os.path.join(fig_prefix, 'psl_{}.pdf'.format(decompose))
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
    print('fig saved in :{}'.format(fig_name) )
    
    
    # show plot
    plt.show()

    
    
        
        
        
        
        
        
    
        
    
    