from reproblem import RE21, RE22, RE23
import numpy as np
from numpy import array
from matplotlib import pyplot as plt
import torch
from torch import Tensor
# problem = RE22()
# problem.evaluate( array([10,10,10]) )
# print()


# pf = np.loadtxt('D:\\code\\reproblems-master\\reproblems-master\\approximated_Pareto_fronts\\reference_points_RE21.dat')
# plt.scatter(pf[:,0], pf[:,1])
# plt.show()

a = torch.ones((10,2))*2


a = torch.ones((10,2))

lb = Tensor([0.2, 0.4])
ub = Tensor([1.2, 1.4])




print()