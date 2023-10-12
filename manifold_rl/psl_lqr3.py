import numpy as np
from numpy import array
import torch
from torch import Tensor

from pf_util import LQR, load_real_pf, get_ep_indices
from torch.autograd import Variable
from torch import optim
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
    
from pymoo.indicators.hv import Hypervolume
ref_point = np.array([3.5, 3.5, 3.5])
hv_indicator = Hypervolume(ref_point=ref_point)

from tqdm import tqdm
import os
from torch import nn
import torch.nn.functional as F



class PrefNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.fc
        # self.conv2 = nn.Conv2d(20, 20, 5)
        self.fc1 = nn.Linear(3, 64)  # 6*6 from image dimension
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = (F.tanh(self.fc3(x))-1)/2
        return x
    
    
    
        
    

def get_theta(r, t):
    k1_1 = -1./(1+torch.exp(r[0] + r[1]*t[:,0] + r[2]*t[:,1]))
    k2_2 = -1./(1+torch.exp(r[3] + r[4]*t[:,0] + r[5]*t[:,1]))
    k3_3 = -1./(1+torch.exp(r[6] + r[7]*t[:,0] + r[8]*t[:,1]))
    return torch.stack([k1_1, k2_2, k3_3])
    

def get_pref_from_angle(angle):
    pref1 = torch.cos(angle[:,0]).unsqueeze(0)
    pref2 = (torch.sin(angle[:,0]) * torch.cos(angle[:,1])).unsqueeze(0)
    pref3 = (torch.sin(angle[:,0]) * torch.sin(angle[:,1])).unsqueeze(0)
    pref = torch.cat([pref1, pref2, pref3], dim=0).T
    return pref
    
    
def get_simplex_pref_from_angle(angle):
    pref = get_pref_from_angle(angle)
    for i in range( len(pref) ):
        pref[i,:] /= torch.sum(pref[i,:])
    return pref
    
    
def get_J(theta):
    n_subproblems = theta.shape[-1]
    J_array = [0] * n_subproblems
    for idx in range(n_subproblems):
        K = torch.diag(theta[:, idx])
        J = lqr.getJ(K)
        J_array[idx] = J.unsqueeze(0)
    J_array = torch.cat(J_array, dim=0)
    return J_array
    
def mtche_loss(J, pref, ref=torch.Tensor([1.95,1.95,1.95])):
    # ref=torch.Tensor([0, 0, 0])
    m=2
    loss_arr = [0] * len(J)
    for idx, (ji, prefi) in enumerate(zip(J, pref)):
        loss_arr[idx] = torch.max((ji - ref)**m / prefi**m)
    return torch.mean(torch.stack(loss_arr))

    
    
    
    
    

if __name__ == '__main__':
    dim_rho = 9
    lqr = LQR(n_obj=3)
    rho_true = Tensor([-1.69, 2.64, 2.84, 1.07, -0.00, -2.99, 1.20, -3.64, -0.14])
    rho = Variable(torch.rand(dim_rho) / 1000, requires_grad=True)
    
    lr = 1e-5
    optimizer = optim.SGD([rho], lr=lr)
    batch_size = 256
    iter_num = 1
    loss_array = [0] * iter_num
    grad_array = [0] * iter_num
    angle_eps = 0.02
    angle_eps_max = np.pi/2 - angle_eps
    num_0 = 3
    
    for iter in tqdm(range(iter_num)):
        angle_array = torch.rand((batch_size,2)) * (angle_eps_max - angle_eps) + angle_eps
        angle_array[:num_0] = angle_eps
        angle_array[num_0*1:num_0*2] = angle_eps_max
        angle_array[num_0*2:num_0*3] = Tensor([angle_eps, angle_eps_max])
        angle_array[num_0*3:num_0*4] = Tensor([angle_eps_max, angle_eps])
        
        pref_array = get_pref_from_angle(angle_array)
        pref_array_simplex = get_simplex_pref_from_angle(angle_array)
        J = get_J(get_theta(rho, angle_array / (np.pi/2)) )
        loss = mtche_loss(J, pref_array)
        loss_array[iter] = float(loss.detach().numpy()) 
        optimizer.zero_grad()
        loss.backward()
        grad_array[iter] = float(torch.norm(rho.grad).numpy())
        optimizer.step()

    test = False
    if not test:
        plt.subplot(2,1,1)
        plt.plot(loss_array)
        plt.ylabel('mTche Loss')
        
        plt.subplot(2,1,2)
        plt.plot(grad_array)
        plt.ylabel('Grad norm')
        plt.show()
        
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    
    with torch.no_grad():
        print('rho psl:{}'.format(rho.detach().numpy()) )
        print('rho true:{}'.format(rho_true))
        
        
        
        test_num = 300
        ang = torch.rand((test_num, 2)) * np.pi/2
        pref_simplex = get_simplex_pref_from_angle(ang)
        
        J_array = get_J(get_theta(rho, ang / (np.pi/2))).numpy()
        hv_val = hv_indicator.do(J_array)
        ax.scatter(J_array[:,0], J_array[:,1], J_array[:,2], label='Estimated')
        pf_true = load_real_pf(n_obj=3)/100
        hv_val_true = hv_indicator.do(pf_true)
        
        solutions_all = np.concatenate((J_array,pf_true),axis=0)
        pf_solutions_all = solutions_all[get_ep_indices(solutions_all)]
        ax.scatter(pf_solutions_all[:,0], pf_solutions_all[:,1], pf_solutions_all[:,2], label='Dominated')
        
        
        
        
        
        ax.scatter(pf_true[:,0], pf_true[:,1], pf_true[:,2], label='True')
        
        print('len Estimated:{}'.format(len(J_array)))
        print('len True:{}'.format(len(pf_true)))
        
        ax.set_xlabel('$J_1$')
        ax.set_ylabel('$J_2$')
        ax.set_zlabel('$J_3$')
        print('hv val:{:.2f}'.format(hv_val))
        print('hv true val:{:.2f}'.format(hv_val_true))
        
        
        
    plt.legend()
    problem_name = 'lqr3'
    fig_prefix = 'C:\\Users\\xzhang2523\\Desktop\\IJCAI_submit\\HV_PSL\\Figures\\hv_approximation'
    fig_prefix = os.path.join(fig_prefix, problem_name)
    os.makedirs(fig_prefix, exist_ok=True)
    mtd = 'psl'
    fig_name = os.path.join(fig_prefix, '{}.pdf'.format(mtd))
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
    print('fig saved in :{}'.format(fig_name) )
    plt.show()
    
    



        