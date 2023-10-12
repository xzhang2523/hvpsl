from mo_optimizers.hv_maximization import HvMaximization
import numpy as np
from numpy import array



n_mo_sol = 3
n_mo_obj = 2
ref = np.zeros(n_mo_obj)
model = HvMaximization(n_mo_sol, n_mo_obj, ref)
weights = model.compute_weights(array([[3,1],[2,2], [1,3]]) )




print()
