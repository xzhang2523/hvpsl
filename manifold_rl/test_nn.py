import torch
from torch import Tensor


from psl_lqr3 import PrefNet

pref = Tensor([0.2,0.2,0.6])
model = PrefNet()

solution = model(pref)
# print()
