import numpy as np
from numpy import array
from matplotlib import pyplot as plt
import torch
from torch import Tensor
from torch.autograd.functional import jacobian
import scipy
import scipy.io
from torch.autograd import Variable

from pf_util import load_real_pf


def element_wise_division(J, Aup=array([1,2])):
    length = J.shape[-1]
    J_new = [0] * length
    for i in range(length):
        res = J[:,i] / Aup
        J_new[i] = res
        # print()
    return torch.stack(J_new) 
        
def get_indicator(J, beta=0.5):
    AUp = Tensor(array([350, 350])) 
    Up = Tensor(array([350, 350])) 
    I_antiutopia = torch.sum((element_wise_division(J, AUp) - torch.ones(2))**2)
    I_utopia = torch.sum( (element_wise_division(J, AUp) - torch.ones(2))**2 )
    I = I_antiutopia * (1 - I_utopia)
    return I
    
    
    
    
def get_theta(r, t):
    # t = torch.linspace(0,1,100)
    k1_1 = -1./(1 + torch.exp(r[0]+r[1]*t))
    k2_2 = -1./(1 + torch.exp(r[2]+r[3]*t))
    return torch.stack([k1_1, k2_2])


    
def get_L():
    
    print()



class LQR:
    def __init__(self, n_obj):
        self.e = 0.1
        self.g = 0.9
        
        self.A = torch.eye(n_obj)
        self.B = torch.eye(n_obj)
        self.E = torch.eye(n_obj)
        self.S = torch.zeros(n_obj)
        self.Sigma = torch.eye(n_obj)
        self.x0 = 10 * torch.ones((n_obj, 1))
        
        self.Q0 = torch.eye(n_obj) * self.e
        self.R0 = torch.eye(n_obj) * (1-self.e)
        self.Q0[0][0] = 1-self.e
        self.R0[0][0] = self.e
        
        self.Q1 = torch.eye(n_obj) * self.e
        self.R1 = torch.eye(n_obj) * (1-self.e)
        self.Q1[1][1] = 1-self.e
        self.R1[1][1] = self.e
        self.Q = [self.Q0, self.Q1]
        self.R = [self.R0, self.R1]
        self.n_obj = n_obj
        
        
        
    def getJ(self, K):
        P = [0,0]
        J = [0,0]
        
        for idx in range(2):
            P[idx] = (self.Q[idx] + K@self.R0[idx]@K) @ torch.inverse(torch.eye(self.n_obj) - self.g*(torch.eye(self.n_obj) + 2*K + K**2))
            J[idx] = self.x0.T @ P[idx] @ self.x0 + (1/(1-g)) * torch.trace(self.Sigma @ (self.R[0] + self.g * self.B.T @ P[idx] @ self.B))
        return torch.cat(J).squeeze()
        
        
        
     
def get_grad():
    pass
    
    
        
if __name__ == '__main__':
    
    # def exp_reducer(x):
        # return x.exp().sum(dim=1)
    # inputs = torch.rand(2, 2)
    # jacobian(exp_reducer, inputs)

    
    # solutions = np.random.random((2,10)) * 200 + 150
    solutions = torch.rand((2,10)) * 200 + 150
    # element_wise_division(solutions)
    
    res = get_indicator(solutions, beta=0.5)
    print()
    
    
    
    
    
    
    lqr = LQR(n_obj=2)
    g = lqr.g
    n_subproblems = 10
    
    rho = Variable(Tensor([1, 2, 0, 3]), requires_grad=True)
    t = Variable(torch.linspace(0, 1, n_subproblems), requires_grad=True) 
    
    D_t_theta = [0]*2
    theta = get_theta(rho, t)
    
    # print()
        
    
    
    
    
    
    # n_subproblems = theta.size()[-1] 
    J_array = [0] * n_subproblems
    for idx in range(n_subproblems):
        K = torch.diag(theta[:, idx])
        J = lqr.getJ(K)
        J_array[idx] = J.unsqueeze(0)
        
    
    
    J_array = torch.cat(J_array, dim=0)
    for idx in range(n_subproblems):
        D_t_theta = [0]*2
        for i in range(2):
            t.grad = None
            theta[i, idx].backward(retain_graph=True)
            D_t_theta[i] = torch.clone(t.grad[i])
        D_t_theta = torch.stack(D_t_theta)
        
        
    
    
        grad_arr = [0]*2
        for obj_idx in range(2):
            J = J_array[idx, obj_idx]
            t.grad = None
            J.backward(retain_graph=True)
            grad_arr[obj_idx] = t.grad[idx]
        
        
        I = get_indicator( J_array[idx,:].unsqueeze(0) )
        
        
        
        T = torch.stack(grad_arr)
        X = T.T @ T
        detX = X
        invX = 1 / detX
        Vol_T = torch.sqrt(detX)
        
        I.backward(retain_graph=True)
        D_rho_I = torch.clone(rho.grad)
        diff1 = Vol_T * D_rho_I
        
        Kr_1 = T.T
        Kr_2 = torch.kron(D_t_theta, torch.eye(2))
        
        
        diff2 = I * Vol_T
        
        
        # Compute Kr2, 3
        # diff2 = I * Vol_T
        
        
        
        
        
        print()
            
            
            
            
    
    
    
    J_array_numpy = torch.cat(J_array, dim=0).numpy()
    print()
    
    
    plt.scatter(J_array[:,0], J_array[:,1], label='learned')
    pf_true = load_real_pf()
    plt.scatter(pf_true[:,0], pf_true[:,1], label='true')
    
    
    plt.legend()
    plt.show()
    print()
    
    
    
    # t = np.linspace(0,1,100)
    # r = array([1,1.1,2,2])
    # k1_1 = -1./(1+np.exp(r[0]+r[1]*t))
    # k2_2 = -1./(1+np.exp(r[2]+r[3]*t))
    # plt.plot(k1_1)
    # plt.plot(k2_2)
    # plt.show()
    
    
    
    
    
    
    # plt.show()
    
    # print()
        
        
    
    