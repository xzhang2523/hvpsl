import torch
import numpy as np
from numpy import array

from problem import loss_function
from reproblem import RE21, RE24, RE37, RE34, RE33
from torch import Tensor





def generate_rand_pref(n_obj, batch_size):
    if n_obj == 2:
        theta = torch.rand(batch_size) * np.pi/2
        pref = torch.cat([torch.sin(theta).unsqueeze(1), torch.cos(theta).unsqueeze(1)], axis=1)
    else:
        theta = torch.rand((batch_size,2)) * np.pi/2
        th1 = theta[:,0]
        th2 = theta[:,1]
        p1 = torch.sin(th1) * torch.sin(th2)
        p2 = torch.sin(th1) * torch.cos(th2)
        p3 = torch.cos(th1)
        pref = torch.cat([p1.unsqueeze(0), p2.unsqueeze(0), p3.unsqueeze(0)], axis=0).T
    return theta, pref



def get_problem(problem_name):
    if problem_name.startswith('RE'):
        problem_dict = {
            'RE21' : RE21(),
            'RE24' : RE24(),
            'RE37' : RE37(),
            'RE34' : RE34(),
            'RE33' : RE33(),
        }
        problem = problem_dict[ problem_name ]
    else:
        problem = None

    return problem



# scipy

def objective(args, x):
    problem = get_problem( args.problem_name )

    if args.problem_name.startswith('RE'):
        J = problem.evaluate(x)
    else:
        J = loss_function(x, problem=args.problem_name )

    return J


def element_wise_division(J, Aup=array([1, 2])):
    length = J.shape[-1]
    J_new = [0] * length
    for i in range(length):
        res = J[:, i] / Aup
        J_new[i] = res
    return torch.stack(J_new)


def get_theta(r, t):
    k1_1 = -1. / (1 + torch.exp(r[0] + r[1] * t))
    k2_2 = -1. / (1 + torch.exp(r[2] + r[3] * t))
    return torch.stack([k1_1, k2_2])


def get_pref_from_angle(angle):
    pref1 = torch.cos(angle).unsqueeze(0)
    pref2 = torch.sin(angle).unsqueeze(0)
    pref = torch.cat([pref1, pref2], dim=0)
    return pref.T



def uniform_sphere_pref(m=2, n=100, eps=1e-2):
    if m == 2:
        theta = np.linspace(eps, np.pi / 2 - eps, n)
        x = np.sin(theta)
        y = np.cos(theta)
        return np.c_[x, y]
    elif m == 3:
        th1 = np.linspace(eps, np.pi / 2 - eps, n)
        th2 = np.linspace(eps, np.pi / 2 - eps, n)
        th1, th2 = np.meshgrid(th1, th2)
        th_array = []
        for i in range(n):
            for j in range(n):
                th_array.append(array([th1[i][j], th2[i][j]]))
        th_array = array(th_array)
        p1 = np.sin(th_array[:, 0]) * np.sin(th_array[:, 1])
        p2 = np.sin(th_array[:, 0]) * np.cos(th_array[:, 1])
        p3 = np.cos(th_array[:, 0])

        return np.c_[p1, p2, p3]



def add_extreme_pref(pref, args=None):
    if args.n_obj == 2:
        pref[0,:] = Tensor([1,0])
        pref[1,:] = Tensor([0,1])
    elif args.n_obj == 3:
        pref[0, :] = Tensor([1, 0, 0])
        pref[1, :] = Tensor([0, 1, 0])
        pref[2, :] = Tensor([0, 0, 1])
    else:
        assert False, 'n_obj should be 2 or 3'
    return pref



from indicators import get_ind_range, get_ind_sparsity
def model_quality(model, args):
    if args.n_obj == 2:
        pref = Tensor(uniform_sphere_pref(m=args.n_obj, n=60))
    else:
        pref = Tensor(uniform_sphere_pref(m=args.n_obj, n=80))

    with torch.no_grad():
        x = model(pref)
        J = objective(args, x).numpy()

        range_val = np.round(get_ind_range(J, args), 2)

        if args.n_obj == 2:
            sparsity_val = get_ind_sparsity(J) * 1e3
        else:
            sparsity_val = get_ind_sparsity(J) * 1e7

    # quality = range_val + sparsity_val
    # quality = range_val

    # quality = range_val - 0.02 * sparsity_val
    quality = sparsity_val
    return quality
