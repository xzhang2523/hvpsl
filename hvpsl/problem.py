import torch
from torch import Tensor
import numpy as np
from pf_util import LQR
from numpy import array







def loss_function(x, problem='zdt1'):
    n_var = x.shape[-1]
    if problem.startswith('zdt'):
        f = x[:, 0]
        g = x[:, 1:]
        
        if problem == 'zdt1':
            g = torch.sum(g, axis=1)*9 / (n_var-1) + 1
            h = 1. - torch.sqrt(f/g)
            
        if problem == 'zdt2':
            g = g.sum(dim=1, keepdim=False) * (9./(n_var-1)) + 1.
            h = 1. - (f/g)**2

        if problem == 'zdt3':
            g = g.sum(dim=1, keepdim=False) * (9./(n_var-1)) + 1.
            h = 1. - torch.sqrt(f/g) - (f/g)*torch.sin(10.*np.pi*f)
        return torch.stack([f, g*h]).T

    elif problem in ['lqr2', 'lqr3']:
        n_subproblems = len(x) 
        J_array = [0] * n_subproblems
        for idx in range(n_subproblems):
            K = torch.diag(x[idx, :])
            if problem == 'lqr2':
                lqr = LQR(n_obj=2)
            else:
                lqr = LQR(n_obj=3)
            J = lqr.getJ(K)
            J_array[idx] = J.unsqueeze(0)
        J_array = torch.cat(J_array, dim=0)
        return J_array
    elif problem == 'vlmop2':
        coeff = torch.sqrt(Tensor([1/n_var]))
        # dx = np.linalg.norm(x + 1. / np.sqrt(n))
        # return 1 - np.exp(-dx**2)
        dx1 = torch.norm(x-coeff, dim=1)
        f1 = 1 - torch.exp(-dx1**2)
        dx2 = torch.norm(x+coeff, dim=1)
        f2 = 1 - torch.exp(-dx2**2)
        # f1 = 1 - torch.exp(- torch.sum(x - torch.sqrt(Tensor([1/30])))**2)
        # f2 = 1 - torch.exp(- torch.sum(x + torch.sqrt(Tensor([1/30])))**2)
        # return torch.stack([f1.unsqueeze(0), f2.unsqueeze(0)])
        res = torch.stack([f1, f2])
        return res.T
    
    elif problem == 'vlmop1':
        f1 = torch.norm(x, dim=1)**2 / n_var / 4
        f2 = torch.norm(x-2, dim=1)**2 / n_var / 4
        res = torch.stack([f1, f2])
        return res.T
    
    
    elif problem.startswith('dtlz2'):
        xm = x[:,2:]
        g = torch.sum( (xm-0.5)**2 ) 
        f1 = (1+g) * torch.cos( np.pi/2*x[:,0] ) * torch.cos(np.pi/2*x[:,1])
        f2 = (1+g) * torch.cos( np.pi/2*x[:,0] ) * torch.sin(np.pi/2*x[:,1])
        f3 = (1+g) * torch.sin( np.pi/2*x[:,0] )
        return torch.stack([f1,f2,f3]).T
    elif problem.startswith('maf1'):
        # Here we consider 3 objective optimization
        # g=0
        xm = x[:,2:]
        g = torch.sum( (xm-0.5)**2 ) 
        f1 = (1+g) * (1-x[:,0]*x[:,1])
        f2 = (1+g) * (1-x[:,0] + x[:,0]*x[:,1])
        f3 = (1+g) * x[:,0]
        return torch.stack([f1,f2,f3]).T
    elif problem == 'vlmop3':
        # 2 -> 3
        print()
        norm_2 = torch.norm( x.squeeze())**2
        f1 = 0.5 * norm_2 + torch.sin(norm_2)

    elif problem == 'vlmop1_m4':
        f1 = torch.norm(x, dim=1) ** 2 / n_var / 4
        f2 = torch.norm(x-1, dim=1) ** 2 / n_var / 4
        f3 = torch.norm(x-2, dim=1) ** 2 / n_var / 4
        f4 = torch.norm(x-1, dim=1) ** 2 / n_var / 4
        print()

        res = torch.stack([f1, f2])


    else:
        assert False, 'problem not defined'
        
def get_pf(problem_name, points_num=100):
    if problem_name == 'vlmop1':
        t = np.linspace(0,2,points_num)
        pf1 = t**2 / 4
        pf2 = (t-2)**2 / 4
        return np.c_[pf1, pf2]
    elif problem_name == 'vlmop2':
        n_var = 10
        coeff = np.sqrt(1/n_var )
        # dx = np.linalg.norm(x + 1. / np.sqrt(n))
        # return 1 - np.exp(-dx**2)
        x = np.linspace(-coeff, coeff, 20 )
        x = np.tile(x, (n_var,1))
        
        
        
        dx1 = np.linalg.norm(x-coeff, axis=1)
        pf1 = 1 - np.exp(-dx1**2)
        dx2 = np.linalg.norm(x+coeff, axis=1)
        pf2 = 1 - np.exp(-dx2**2)
        
        return np.c_[pf1, pf2]
        
        
    elif problem_name == 'zdt1':
        t = np.linspace(0, 1, points_num)
        pf1 = t
        pf2 = 1-np.sqrt(t)
        return np.c_[pf1, pf2]
        
    elif problem_name == 'dtlz2':
        num = 15
        th1 = np.linspace(0,np.pi/2, num)
        th2 = np.linspace(0,np.pi/2, num)
        thth1, thth2 = np.meshgrid(th1, th2)
        res = []
        for i in range(num):
            for j in range(num):
                th1 = thth1[i]
                th2 = thth2[i]
                aa = np.array([np.cos(th1), np.sin(th1) * np.cos(th2), np.sin(th1) * np.sin(th2)])
                res.append(aa)
            
        # x = np.cos(thth1)
        # y = np.sin(thth1) * np.cos(thth2)
        # z = np.sin(thth1) * np.sin(thth2)
        return np.array(res)
    
    elif problem_name == 'maf1':
        res = np.array([[0,0,0]])
        return res
        # def sphere(th1, th2):
            
        
        
    
    
def get_true_hv(problem_name):
    if problem_name == 'vlmop1':
        return 5/6
    elif problem_name == 'vlmop2':
        return 0.5
    elif problem_name == 'zdt1':
        return 2/3
    
    
    


        
    
        
    
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    
    solutions = uniform_sphere_pref(m=3)
    
    ax = plt.axes(projection='3d')
    # plt.scatter(solutions[:,0], solutions[:,1])
    ax.scatter3D(solutions[:,0], solutions[:,1], solutions[:,2], color = "green", label='PSL')
    plt.show()
    # print()
    