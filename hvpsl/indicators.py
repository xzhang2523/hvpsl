import numpy as np
from numpy import array
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import Hypervolume





def get_ind_sparsity(obj_batch):
    obj_num = obj_batch.shape[-1]
    non_dom = NonDominatedSorting().do(obj_batch, only_non_dominated_front=True)
    objs = obj_batch[non_dom]

    if obj_num == 2:
        sort_array = obj_batch
        sparsity = [np.linalg.norm(sort_array[i] - sort_array[i + 1]) ** 2 for i in range(len(sort_array) - 1)]
        sparsity = np.max(sparsity)
    else:
        sparsity_sum = 0
        for objective in range(objs.shape[-1]):
            objs_sort = np.sort(objs[:, objective])
            sp = 0
            for i in range(len(objs_sort) - 1):
                sp += np.power(objs_sort[i] - objs_sort[i + 1], 2)
            sparsity_sum += sp
        if len(objs) > 1:
            sparsity = sparsity_sum / (len(objs) - 1)
        else:
            sparsity = 0
    return sparsity


def get_ind_range(J):
    x = J[:,0]
    y = J[:,1]
    z = J[:,2]
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arctan(np.sqrt(x**2+y**2)/ np.clip(z**2, np.ones_like(x)*1e-3, a_max=None) )
    fie = np.arctan(y / np.clip(x, np.ones_like(x)*1e-3, a_max=None) )
    theta_range = np.max(theta) - np.min(theta)
    fie_range = np.max(fie) - np.min(fie)
    return min(theta_range, fie_range)



def get_ind_hv(args):
    hv_indicator = Hypervolume( ref_point=np.ones(args.n_obj)*3.5 )
    return hv_indicator



