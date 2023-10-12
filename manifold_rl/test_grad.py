from torch.autograd import Variable
from torch import Tensor
import torch

x = Variable(Tensor([2.0]), requires_grad=True)

y = x**2

torch.clip(torch.sum(y),0,3).backward()
print()
