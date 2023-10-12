import numpy as np

import sys
sys.path.append('D:\\code\\Paper_IJCAI\\HypervolumeDerivatives-main')

from hvd import HypervolumeDerivatives
from hvd.utils import get_non_dominated

from matplotlib import pyplot as plt


def MOP1(x):
    x = np.array(x)
    return np.array([np.sum((x - 1) ** 2), np.sum((x + 1) ** 2)])


def MOP1_Jacobian(x):
    x = np.array(x)
    return np.array([2 * (x - 1), 2 * (x + 1)])


def MOP1_Hessian(x):
    x = np.array(x)
    return np.array([2 * np.eye(2), 2 * np.eye(2)])


ref = np.array([20, 20])
hvh = HypervolumeDerivatives(
    dim_d=2, dim_m=2, ref=ref, func=MOP1, jac=MOP1_Jacobian, hessian=MOP1_Hessian, minimization=False
)

iterations = 1000
for i in range(iterations):
    X = np.random.rand(10, 2)
    Y = np.array([MOP1(_) for _ in X])
    idx = get_non_dominated(-1 * Y, return_index=True)
    out = hvh.compute_hessian(X[idx, :])
    HdY2, HdX2 = out["HVdY2"], out["HVdX2"]
    HdX = out["HVdX"]
    w, v = np.linalg.eigh(HdX2)
    # print(w)




print('shape,{}'.format(Y.shape))
plt.scatter(Y[:,0], Y[:,1])
plt.show()