import torch
import numpy as np

from problem import loss_function
from reproblem import RE21, RE24, RE37, RE34, RE33




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
