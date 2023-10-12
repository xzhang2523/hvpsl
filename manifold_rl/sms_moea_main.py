from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.visualization.scatter import Scatter
from reproblem import RE37, RE24, RE21

import numpy as np
from pymoo.core.problem import Problem
import time

import torch
from matplotlib import pyplot as plt
import os

from pymoo.indicators.hv import Hypervolume
from problem import loss_function


reproblem_dict = {'RE37': RE37(), 'RE24': RE24(), 'RE21': RE21()}



class MyProblem(Problem):
    def __init__(self, problem_name):
        # self.n_objectives = 3
        # self.n_variables = 4
        self.problem_name = problem_name
        self.reproblem = reproblem_dict[problem_name]
        
        if problem_name.startswith('RE'):
            n_var = self.reproblem.n_variables
            n_obj = self.reproblem.n_objectives
            lb = self.reproblem.lbound.numpy()
            ub = self.reproblem.ubound.numpy()
            
        else:
            n_var = 5
            n_obj = 2
            lb = 0
            ub = 1
        super().__init__(n_var = n_var,
                         n_obj = n_obj,
                         xl=lb,
                         xu=ub)

    def _evaluate(self, x, out, *args, **kwargs):
        
        # f1 = 100 * (x[:, 0]**2 + x[:, 1]**2)
        # f2 = (x[:, 0]-1)**2 + x[:, 1]**2
        if self.problem_name.startswith('RE'):
            re_res = self.reproblem.evaluate( torch.Tensor(x) )
        else:
            re_res = loss_function( torch.Tensor(x), problem=self.problem_name )
            
        out["F"] = re_res.numpy()
        


if __name__ == '__main__':

    # problem_name = 'RE24'
    # if problem_name == 'RE37':
    #     re_problem = RE37()
    # elif problem_name == 'RE24':
    #     re_problem = RE24()
    # elif problem_name == 'RE21':
    #     re_problem = RE21()
    problem_name = 'RE37'
    
    n_obj_dict = {'RE37': 3, 'RE24': 2, 'RE21': 2, 'vlmop2': 2, 'zdt1': 2}
    n_obj = n_obj_dict[problem_name]
    problem = MyProblem(problem_name=problem_name)
    
    
    
    pop_size = 40 if n_obj==2 else 100
    algorithm = SMSEMOA(pop_size=pop_size)
    
    nGeneration = 100
    
    start = time.time()
    res = minimize(problem,
                algorithm,
                ('n_gen', nGeneration),
                seed=1,
                verbose=False)
    
    
    
    hv_indicator = Hypervolume(ref_point=np.ones(n_obj)*3.5)
    hv_val = hv_indicator.do(res.F)
    
    elapse = time.time() - start
    
    print("Time: {:.2f}".format(elapse / 60))
    print("HV: {:.2f}".format(hv_val))
    
    
    
    if n_obj == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.view_init(elev=46, azim=33)
        ax.scatter3D(res.F[:,0], res.F[:,1], res.F[:,2], color = "green", label='SMS-EMOA')
        ax.set_xlabel('$f_1(x)$', fontsize=16)
        ax.set_ylabel('$f_2(x)$', fontsize=16)
        ax.set_zlabel('$f_3(x)$', fontsize=16)
    else:
        plt.scatter(res.F[:,0], res.F[:,1], color = "gold")
        plt.xlabel('$f_1(x)$', fontsize=16)
        plt.ylabel('$f_2(x)$', fontsize=16)



    fig_folder = os.path.join('C:\\Users\\xzhang2523\\Desktop\\IJCAI_submit\\HV_PSL\\Figures', 'smsmoea')
    os.makedirs(name=fig_folder, exist_ok=True)
    fig_name = os.path.join(fig_folder, '{}.pdf'.format(problem_name))
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
    print('Figure saved to {}'.format(fig_name))
    
    plt.show()
    



    # plot = Scatter()
    # plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    # plot.add(res.F, color="red")
    
    
    
    # plot.show()