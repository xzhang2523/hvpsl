import torch.nn.functional as F
from torch import nn



class PrefNet(nn.Module):
    def __init__(self, args, problem=None):
        super().__init__()
        self.problem_name = args.problem_name
        if self.problem_name.startswith('RE'):
            self.lb = problem.lbound
            self.ub = problem.ubound
            n_var = problem.n_variables
            hidden_size = 256
            
        else:
            n_var = args.n_var
            hidden_size = 64
            
        self.fc1 = nn.Linear(args.n_obj, hidden_size)  # 6*6 from image dimension
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, n_var)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        if self.problem_name in ['zdt1', 'zdt2', 'dtlz2']:
            x = F.sigmoid(self.fc4(x))  
        elif self.problem_name in ['lqr2', 'lqr3']:
            x = F.sigmoid(self.fc4(x)) - 1
        elif self.problem_name.startswith('RE'):
            x = F.sigmoid(self.fc4(x)) * (self.ub - self.lb) + self.lb
        elif self.problem_name in ['vlmop1', 'vlmop2']:
            x = self.fc4(x)
        else:
            assert False
        return x