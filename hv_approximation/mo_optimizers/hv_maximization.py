"""
The class HvMaximization is based on the algorithm described by
Wang, Hao, et al.
"Hypervolume indicator gradient ascent multi-objective optimization."
International conference on evolutionary multi-criterion optimization. Springer, Cham, 2017.
"""

import numpy as np
import torch
from torch import Tensor

import sys
sys.path.append('D:\code\Paper_IJCAI\hv_moo\mo_optimizers')
from functions_evaluation import fastNonDominatedSort
from functions_hv_grad_3d import grad_multi_sweep_with_duplicate_handling
from pymoo.indicators.hv import Hypervolume


class HvMaximization(object):
    """
    Mo optimizer for calculating dynamic weights using higamo style hv maximization
    based on Hao Wang et al.'s HIGA-MO
    uses non-dominated sorting to create multiple fronts, and maximize hypervolume of each
    """
    def __init__(self, n_mo_sol, n_mo_obj, ref_point, obj_space_normalize=True):
        super(HvMaximization, self).__init__()
        self.name = 'hv_maximization'
        self.ref_point = np.array(ref_point)
        self.n_mo_sol = n_mo_sol
        self.n_mo_obj = n_mo_obj
        self.obj_space_normalize = obj_space_normalize


    def compute_weights(self, mo_obj_val):
        n_mo_obj = self.n_mo_obj
        n_mo_sol = self.n_mo_sol

        # non-dom sorting to create multiple fronts
        hv_subfront_indices = fastNonDominatedSort(mo_obj_val)
        dyn_ref_point =  1.1 * np.max(mo_obj_val, axis=1)
        for i_obj in range(0,n_mo_obj):
            dyn_ref_point[i_obj] = np.maximum(self.ref_point[i_obj],dyn_ref_point[i_obj])
        number_of_fronts = np.max(hv_subfront_indices) + 1 # +1 because of 0 indexing
        
        
        # number_of_fronts = 1
        obj_space_multifront_hv_gradient = np.zeros((n_mo_obj,n_mo_sol))
        for i_fronts in range(0, number_of_fronts):
            # compute HV gradients for current front
            temp_grad_array = grad_multi_sweep_with_duplicate_handling(mo_obj_val[:, (hv_subfront_indices == i_fronts) ],dyn_ref_point)
            obj_space_multifront_hv_gradient[:, (hv_subfront_indices == i_fronts) ] = temp_grad_array

        # normalize the hv_gradient in obj space (||dHV/dY|| == 1)
        normalized_obj_space_multifront_hv_gradient = np.zeros((n_mo_obj,n_mo_sol))
        for i_mo_sol in range(0,n_mo_sol):
            w = np.sqrt(np.sum(obj_space_multifront_hv_gradient[:,i_mo_sol]**2.0))
            if np.isclose(w,0):
                w = 1
            if self.obj_space_normalize:
                normalized_obj_space_multifront_hv_gradient[:,i_mo_sol] = obj_space_multifront_hv_gradient[:,i_mo_sol]/w
            else:
                normalized_obj_space_multifront_hv_gradient[:,i_mo_sol] = obj_space_multifront_hv_gradient[:,i_mo_sol]

        dynamic_weights = torch.tensor(normalized_obj_space_multifront_hv_gradient, dtype=torch.float)
        return(dynamic_weights)




class HvMaximizationRaw(object):
    def __init__(self, n_mo_sol, n_mo_obj, ref_point, obj_space_normalize=True):
        super(HvMaximizationRaw, self).__init__()
        self.name = 'hv_maximization_raw'
        self.ref_point = np.array(ref_point)
        self.n_mo_sol = n_mo_sol
        self.n_mo_obj = n_mo_obj
        self.obj_space_normalize = obj_space_normalize
        self.hv = Hypervolume(ref_point=ref_point)
        
    def num_grad(self, mo_obj):
        hv0 = self.hv.do(mo_obj)
        hv_grad = np.zeros_like(mo_obj)
        m,n = mo_obj.shape
        eps = 1e-3
        for i in range(m):
            for j in range(n):
                mo_obj_i = np.copy(mo_obj)
                mo_obj_i[i,j] += eps
                hv_i = self.hv.do(mo_obj_i)
                hv_grad[i,j] = (hv_i - hv0) / eps
                
        return hv_grad
        # print()
        
        
        
    def compute_weights(self, mo_obj_val):
        mo_obj =np.copy(mo_obj_val).T
        mo_grad_val = self.num_grad(mo_obj)
        return -Tensor(mo_grad_val.T)  
        
        
        
    
        
    



if __name__ == '__main__':
    
    
    def f(x):
        return x[:,0]
    
    
    
    n_mo_sol, n_mo_obj, ref_point = 3,2,np.array([3,3])
    # ref_point = 
    opt = HvMaximization(n_mo_sol, n_mo_obj, ref_point)
    mo_obj_val = np.array([[1,2],[2,1],[1.5,1.5]]).T
    weight = opt.compute_weights( mo_obj_val)
    
    lr = 1e-1
    n_iter = 100
    for _ in range(n_iter):
        weight = opt.compute_weights( mo_obj_val)
        mo_obj_val -= lr * weight.numpy()
        
    
    
    # print()