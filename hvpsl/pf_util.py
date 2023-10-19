import scipy
import torch
import numpy as np
from numpy import array
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting



def load_real_pf(problem_name='zdt1', n_obj=2, down_sample=3):
    if problem_name == 'zdt1':
        x = np.linspace(0,1,100)
        y = 1 - np.sqrt(x)
        pf = np.c_[x,y]
    elif problem_name == 'zdt2':
        x = np.linspace(0,1,100)
        y = 1 - x**2
        pf = np.c_[x,y]
    elif problem_name.startswith('lqr'):
        file_path = 'D:\\code\\Paper_IJCAI\\hvpsl\\data\\front_{}.mat'.format(problem_name)
        mat = scipy.io.loadmat(file_path)
        front_manual = mat['front_manual'] / 100
        pf = front_manual[::down_sample]
    elif problem_name == 'vlmop1':
        t = np.linspace(0, 2, 100)
        x = t**2/4
        y = (t-2)**2/4
        pf = np.c_[x,y]
    elif problem_name == 'vlmop2':
        t = np.linspace(-1, 1, 100)
        x = 1 - np.exp(-(t-1)**2)
        y = 1 - np.exp(-(t+1)**2)
        pf = np.c_[x,y]
    else:
        print('{} not implemeted'.format(problem_name))
        assert False
    return pf


def load_re_pf(problem_name):
    pf = np.loadtxt(
        'D:\\code\\reproblems-master\\reproblems-master\\approximated_Pareto_fronts\\reference_points_{}.dat'.format(
            problem_name))
    return pf


def load_re_pf_norm(problem_name, down_sample=10):
    pf = load_re_pf(problem_name)
    pf_min = np.min(pf, axis=0)
    pf_max = np.max(pf, axis=0)
    for i in range(len(pf)):
        pf[i, :] = (pf[i, :] - pf_min) / (pf_max - pf_min)

    pf = pf[::10]
    idx = np.argsort(pf[:, 0])
    pf = pf[idx]

    return pf









def check_dominated(obj_batch, obj):
    return (np.logical_and(
                (obj_batch <= obj).all(axis=1), 
                (obj_batch < obj).any(axis=1))
            ).any()
    
    
def get_ep_indices(obj_batch_input):
    if len(obj_batch_input) == 0: return np.array([])
    obj_batch = np.array(obj_batch_input)
    sorted_indices = np.argsort(obj_batch.T[0])
    ep_indices = []
    for idx in sorted_indices:
        if (obj_batch[idx] >= 0).all() and not check_dominated(obj_batch, obj_batch[idx]):
            ep_indices.append(idx)
    return ep_indices




    



if __name__ == '__main__':
    print()
    
    
    