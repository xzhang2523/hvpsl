from torch import Tensor

mtche_sgd_lr_dict = {
  'RE21': 1e-1,
  'RE24': 1e-1,
  'RE37': 1,
  'zdt1' : 1e-1,
  'zdt2' : 1e-1,
  'vlmop1' : 1e-1,
  'vlmop2' : 1e-1,
  'lqr2' : 1e-1,
  'lqr3' : 1e-1,
}

tche_sgd_lr_dict = {
  'RE21': 1e-1,
  'RE24': 1e-1,
  'RE37': 0.5,
  'zdt1' : 1e-1,
  'zdt2' : 1e-1,
  'vlmop1' : 1e-1,
  'vlmop2' : 1e-1,
  'lqr2' : 1e-1,
  'lqr3' : 1e-1,
}

epo_sgd_lr_dict = {
  'RE21': 0.1,
  'RE24': 0.1,
  'RE37': 0.5,
  'zdt1' : 0.1,
  'zdt2' : 0.1,
  'vlmop1' : 0.1,
  'vlmop2' : 0.1,
  'lqr2' : 0.1,
  'lqr3' : 0.1,
}

hv1_sgd_lr_dict = {
  'RE21': 0.1,
  'RE24': 0.1,
  'RE37': 0.1,
  'zdt1' : 0.01,
  'zdt2' : 0.01,
  'vlmop1' : 0.05,
  'vlmop2' : 0.05,
  'lqr2' : 0.05,
  'lqr3' : 0.05,
  'vlmop1_m4' : 0.01
}

hv2_sgd_lr_dict = {
  'RE21': 2e-3,
  'RE24': 2e-3,
  'RE37': 0.1,
  'zdt1' : 1e-3,
  'zdt2' : 1e-2,
  'vlmop1' : 1e-2,
  'vlmop2' : 1e-2,
  'lqr2' : 1e-2,
  'lqr3' : 1e-2,
}



ideal_point_dict = {
  'RE21': Tensor([0.0, 0.0]),
  'RE24': Tensor([0.0, 0.0]),
  'RE37': Tensor([0.0, 0.0, 0.0]),
  'zdt1' : Tensor([0.0, 0.0]),
  'zdt2' : Tensor([0.0, 0.0]),
  'dtlz2' : Tensor([0.0, 0.0, 0.0]),
  'vlmop1' : Tensor([0.0, 0.0]),
  'vlmop2' : Tensor([0.0, 0.0]),
  'lqr2' : Tensor([1.50, 1.50]),
  'lqr3' : Tensor([1.9, 1.9, 1.9]),
}


nadir_eps = 0
nadir_point_dict = {
  'RE21': Tensor([1.0, 1.0]) + nadir_eps,
  'RE24': Tensor([1.0, 1.0]) + nadir_eps,
  'RE37': Tensor([1.05, 1.05, 1.05]),
  'dtlz2': Tensor([1.2, 1.2, 1.2]),
  'zdt1' : Tensor([1.0, 1.0]) + nadir_eps,
  'zdt2' : Tensor([1.0, 1.0]) + nadir_eps,
  'vlmop1' : Tensor([1.0, 1.0]) + nadir_eps,
  'vlmop2' : Tensor([1.0, 1.0]) + nadir_eps,
  'lqr2' : Tensor([4.0, 4.0]),
  'lqr3' : Tensor([4.0, 4.0, 4.0]),
}

sparse_scale_dict = {
  2 : 1e3,
  3 : 1e7,
}


clip_norm_dict = {
  'mtche' : 5.0,
  'hv2' : 2.0,
}







def get_nobj(problem):
  pass


