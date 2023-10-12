from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize
from pymoo.problems import get_problem
# from pymoo.visualization.scatter import Scatter
from matplotlib import pyplot as plt
import os
from time import time

problem = get_problem("dtlz2")
algorithm = SMSEMOA()


FONT_SIZE = 16

ts = time()
res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               seed=1,
               verbose=False)
elapse = (time() - ts)/60
print('elapse:{:.2f}'.format(elapse))

ax = plt.axes(projection ="3d")
ax.view_init(elev=46, azim=33)
ax.scatter3D(res.F[:,0], res.F[:,1], res.F[:,2], color = "green")
ax.set_xlabel('$f_1(x)$', fontsize=FONT_SIZE)
ax.set_ylabel('$f_2(x)$', fontsize=FONT_SIZE)
ax.set_zlabel('$f_3(x)$', fontsize=FONT_SIZE)
fig_name = os.path.join('C:\\Users\\xzhang2523\\Desktop\\IJCAI rebuttal', 'sms_moea.pdf') 
plt.savefig(fig_name, bbox_inches='tight')
plt.show()
print('saved in {}'.format(fig_name))
# plt.scatter(x, y)

# plot = Scatter()
# plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
# plot.add(res.F, color="red")
# plot.show()