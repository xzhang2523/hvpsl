import torch
from torch import Tensor
import numpy as np


def loss_function(x, problem='zdt1'):
    n_var = x.shape[-1]
    if problem.startswith('zdt'):
        f = x[:, 0]
        g = x[:, 1:]
        
        if problem == 'zdt1':
            g = torch.sum(g, axis=1)*9/(n_var-1) + 1
            h = 1. - torch.sqrt(f/g)
            
        if problem == 'zdt2':
            g = g.sum(dim=1, keepdim=False) * (9./(n_var-1)) + 1.
            h = 1. - (f/g)**2

        if problem == 'zdt3':
            g = g.sum(dim=1, keepdim=False) * (9./(n_var-1)) + 1.
            h = 1. - torch.sqrt(f/g) - (f/g)*torch.sin(10.*np.pi*f)
            
        return torch.stack([f,g*h])
    
    
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
        
        return torch.stack([f1, f2])
    elif problem == 'vlmop3':
        x1 = x[:,0]
        x2 = x[:,1]
        f1 = 0.5 * (x1**2+x2**2) + torch.sin(x1**2+x2**2)
        f2 = (3*x1 - 2*x2 + 4)**2/8 + (x1-x2+1)**2/27 + 15
        f3 = 1/(x1**2+x2**2+1) - 1.1*torch.exp(-x1**2-x2**2)
        return torch.stack([f1, f2,f3])

        
    
    
    elif problem == 'vlmop1':
        f1 = torch.norm(x, dim=1)**2 / n_var / 4
        f2 = torch.norm(x-2, dim=1)**2 / n_var / 4
        return torch.stack([f1, f2])
    
    
    elif problem.startswith('dtlz2'):
        # Here we consider 3 objective optimization
        # g=0
        xm = x[:,2:]
        g = torch.sum( (xm-0.5)**2 ) 
        f1 = (1+g) * torch.cos( np.pi/2*x[:,0] ) * torch.cos(np.pi/2*x[:,1])
        f2 = (1+g) * torch.cos( np.pi/2*x[:,0] ) * torch.sin(np.pi/2*x[:,1])
        f3 = (1+g) * torch.sin( np.pi/2*x[:,0] )
        return torch.stack([f1,f2,f3])
    elif problem.startswith('maf1'):
        # Here we consider 3 objective optimization
        # g=0
        xm = x[:,2:]
        g = torch.sum( (xm-0.5)**2 ) 
        f1 = (1+g) * (1-x[:,0]*x[:,1])
        f2 = (1+g) * (1-x[:,0] + x[:,0]*x[:,1])
        f3 = (1+g) * x[:,0]
        return torch.stack([f1,f2,f3])
    elif problem == 'vlmop3':
        # 2 -> 3
        print()
        norm_2 = torch.norm( x.squeeze())**2
        f1 = 0.5 * norm_2 + torch.sin(norm_2)
    else:
        assert False, 'problem not defined'
        
        
        
        
def get_pf(problem_name, points_num=100):
    if problem_name == 'vlmop1':
        t = np.linspace(0,2,points_num)
        pf1 = t**2 / 4
        pf2 = (t-2)**2 / 4
        return np.c_[pf1, pf2]
    elif problem_name == 'vlmop2':
        n_var = 1
        coeff = np.sqrt(1/n_var )
        x = np.linspace(-coeff, coeff, 20 )
        # x = np.tile(x, (n_var,1))
        
        dx1 = x-coeff
        pf1 = 1 - np.exp(-dx1**2)
        dx2 = x+coeff
        pf2 = 1 - np.exp(-dx2**2)
        
        
        
        return np.c_[pf1, pf2]
        
        
    elif problem_name == 'zdt1':
        t = np.linspace(0, 1, points_num)
        pf1 = t
        pf2 = 1-np.sqrt(t)
        return np.c_[pf1, pf2]
    elif problem_name == 'dtlz2':
        t = np.linspace(0, np.pi/2, 20)
        s = np.linspace(0, np.pi/2, 20)
        t,s = np.meshgrid(t,s)
        x = np.sin(s) * np.cos(t)
        y = np.sin(s) * np.sin(t)
        z = np.sin(s)
        return np.c_[x,y,z]
        
        
        
    
    
def get_true_hv(problem_name):
    if problem_name == 'vlmop1':
        return 5/6
    elif problem_name == 'vlmop2':
        return 0.5
    elif problem_name == 'zdt1':
        return 2/3
    
    
    
        
    