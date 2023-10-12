from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize
from pymoo.problems import get_problem
# from pymoo.visualization.scatter import Scatter
from matplotlib import pyplot as plt
import os
from time import time
import numpy as np

problem = get_problem("dtlz2")

algorithm = SMSEMOA()


ax = plt.axes(projection ="3d")
ax.view_init(elev=46, azim=33)

r = 0.01
u, v = np.mgrid[0:np.pi/2:30j, 0:np.pi/2:20j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
ax.scatter3D(x, y, z)



ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig_name = os.path.join('C:\\Users\\xzhang2523\\Desktop\\IJCAI rebuttal', 'hvpsl.pdf') 
plt.savefig(fig_name, bbox_inches='tight')
plt.show()
print('saved in {}'.format(fig_name))
# plt.scatter(x, y)

# plot = Scatter()
# plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
# plot.add(res.F, color="red")
# plot.show()