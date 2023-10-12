from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.moead import MOEAD

from pymoo.optimize import minimize
from pymoo.problems import get_problem
# from pymoo.visualization.scatter import Scatter
from matplotlib import pyplot as plt
import os
from time import time
from pymoo.util.ref_dirs import get_reference_directions


problem = get_problem("dtlz2")

ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)

algorithm = MOEAD(
    ref_dirs,
    n_neighbors=15,
    prob_neighbor_mating=0.7,
)

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
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig_name = os.path.join('C:\\Users\\xzhang2523\\Desktop\\IJCAI rebuttal', 'moead.pdf') 
plt.savefig(fig_name, bbox_inches='tight')
plt.show()
print('saved in {}'.format(fig_name))
# plt.scatter(x, y)

# plot = Scatter()
# plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
# plot.add(res.F, color="red")
# plot.show()