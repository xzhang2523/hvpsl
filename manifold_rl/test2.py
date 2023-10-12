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

from pf_util import load_real_pf



from pymoo.indicators.hv import Hypervolume
ref_point = np.array([350, 350])
hv_indicator = Hypervolume(ref_point=ref_point)
# print("HV", ind(A))





def element_wise_division(J, Aup=array([1,2])):
    length = J.shape[-1]
    J_new = [0] * length
    for i in range(length):
        res = J[:,i] / Aup
        J_new[i] = res
        # print()
    return torch.stack(J_new) 
        
def get_indicator(J, beta=1.0):
    AUp = Tensor(array([350, 350])) 
    Up = Tensor(array([350, 350])) 
    I_antiutopia = torch.sum((element_wise_division(J, AUp) - torch.ones(2))**2)
    I_utopia = torch.sum( (element_wise_division(J, Up) - torch.ones(2))**2 )
    I = I_antiutopia * (1 - beta * I_utopia)
    return I
    
    
    
    
def get_theta(r, t):
    # t = torch.linspace(0,1,100)
    k1_1 = -1./(1 + torch.exp(r[0]+r[1]*t))
    k2_2 = -1./(1 + torch.exp(r[2]+r[3]*t))
    return torch.stack([k1_1, k2_2])

def load_pf():
    file_path = 'D:\\code\\Paper_IJCAI\\manifold_rl\\front_lqr2.mat'
    mat = scipy.io.loadmat(file_path)
    front_manual = mat['front_manual']
    return front_manual
    
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
    
    
    
def get_J(rho):
    with torch.no_grad():
        t = Variable(torch.linspace(0, 1, n_subproblems), requires_grad=True) 
        theta = get_theta(rho, t)
        J_array = [0] * n_subproblems
        for idx in range(n_subproblems):
            K = torch.diag(theta[:, idx])
            J = lqr.getJ(K)
            J_array[idx] = J.unsqueeze(0)
        J_array = torch.cat(J_array, dim=0)
    return J_array.numpy()
    
    
    
def get_L(rho):
    t = Variable(torch.linspace(0, 1, n_subproblems), requires_grad=True) 
    theta = get_theta(rho, t)
    J_array = [0] * n_subproblems
    for idx in range(n_subproblems):
        K = torch.diag(theta[:, idx])
        J = lqr.getJ(K)
        J_array[idx] = J.unsqueeze(0)
    J_array = torch.cat(J_array, dim=0)
    L = [0] * n_subproblems
    for idx in range(n_subproblems):
        D_t_J = [0]*2
        for obj_idx in range(2):
            J = J_array[idx, obj_idx]
            t.grad = None
            J.backward(retain_graph=True)
            D_t_J[obj_idx] = t.grad[idx]
        I = get_indicator( J_array[idx,:].unsqueeze(0) )
        T = torch.stack(D_t_J)
        X = T.T @ T
        detX = X
        invX = 1 / detX
        Vol_T = torch.sqrt(detX)
        L[idx] = Vol_T * I
    L = torch.stack(L)
    
    L = torch.trapz(L, t)
    
    return L
    
    
def get_diff_rho(rho):
    
    f0 = get_L(rho)
    grad = [0] * len(rho)
    
    eps = 1e-3
    for idx in range(len(rho)):
        rho_tmp = torch.clone(rho)
        rho_tmp[idx] += eps
        
        f_tmp = get_L(rho_tmp)
        grad[idx] = (f_tmp - f0) / eps
    return torch.stack(grad)
    # print()
        
        
        
        
    
    
        
    
if __name__ == '__main__':
    solutions = torch.rand((2,10)) * 200 + 150
    
    res = get_indicator(solutions)
    print()
    
    
    
    
    
    
    lqr = LQR(n_obj=2)
    g = lqr.g
    n_subproblems = 100
    
    rho = Variable(Tensor([1, 2, 0, 3]), requires_grad=True)
    
    
    
    lr = 1e-3
    total_iteration = 10
    for _ in tqdm(range(total_iteration)):
        grad = get_diff_rho(rho)
        grad_ts = grad.detach()
        rho.grad = None
        rho.data += lr * grad_ts
        # print()
        
    
    
    
    J = get_J(rho)
    hv_val = hv_indicator.do(J)
    hv_scalar = 1e4
    print('PGMA hv:{:.2f}'.format(hv_val / hv_scalar))
    
    
    pf_real = load_real_pf()
    hv_true_val = hv_indicator.do(pf_real)
    print('True hv:{:.2f}'.format(hv_true_val / hv_scalar))
    
    
    
    
    plt.scatter(J[:,0], J[:,1], label='PGMA')
    plt.scatter(pf_real[:,0], pf_real[:,1], label='True')
    
    plt.legend()
    plt.show()
    
    
    print()
    
    
    
        
        
        
        
        
        
    
        
    
    