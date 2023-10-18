import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# pymoo

# torch
import torch
from torch import optim

# self-lib
from hvpsl.model.psl_model import PrefNet
from psl_util import generate_rand_pref, objective

# 3rd-lib
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
import warnings
warnings.simplefilter("ignore", UserWarning)
import time

from solvers import EPOSolver
# constants or functions
from moo_data import hv1_sgd_lr_dict, hv2_sgd_lr_dict, mtche_sgd_lr_dict, tche_sgd_lr_dict, epo_sgd_lr_dict, \
    nadir_point_dict, ideal_point_dict






def psl_loss(J, pref, args, x=None, theta=None, nadir_ref=None):
    m = args.n_obj
    loss_arr = [0] * len(J)
    decompose = args.decompose
    for idx, (ji, prefi) in enumerate(zip(J, pref)):
        if decompose == 'tche':
            loss_arr[idx] = torch.max((ji - args.ideal_point+0.25) * prefi )
        elif decompose in ['mtche', 'mtchenoclip']:
            loss_arr[idx] = torch.max((ji - args.ideal_point) / prefi )
        elif decompose in ['hv1', 'hv1noclip']:
            arg_idx = torch.argmin((nadir_ref - ji) / prefi)
            rho = ((nadir_ref - ji) / prefi)[arg_idx]
            if m==2:
                if rho > 0:
                    loss_arr[idx] = -rho**m
                else:
                    loss_arr[idx] = -rho
            else:
                loss_arr[idx] = -rho**m
        elif decompose in ['hv2', 'hv2noclip']:
            if m ==2:
                loss_arr[idx] = torch.max((ji - args.ideal_point)**2 / (prefi**2))
            elif m==3:
                loss_arr[idx] = torch.max((ji - args.ideal_point)**3 / (prefi**3)) 
                
        elif decompose == 'ls':
            loss_arr[idx] = (ji - args.ideal_point) @ prefi
        elif decompose == 'epo':
            solver = EPOSolver(n_tasks=args.n_obj, n_params = args.n_var)
            loss_arr[idx] = solver.get_weighted_loss((ji - args.ideal_point), ray=1/prefi, parameters=x)
    return torch.mean(torch.stack(loss_arr))
    


def main_loop(args, nadir_ref):
    model = PrefNet( args=args)

    if args.optimizer_name=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    for iter in tqdm(range(n_iter)):
        theta, pref = generate_rand_pref(n_obj=args.n_obj, batch_size=args.batch_size)
        pref = torch.clamp(pref, pref_eps, pref_eps_max)
        x = model(pref)
        J = objective(args, x)

        loss = psl_loss(J, pref, args=args, x=x, theta=theta, nadir_ref=nadir_ref)

        optimizer.zero_grad()
        loss.backward()
        loss_array[iter] = loss.detach().numpy()
        if not args.decompose.endswith('noclip'):
            if args.decompose == 'mtche':
                max_norm = 5.0
            else:
                max_norm = 2.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=2.0, error_if_nonfinite=False)
        optimizer.step()


    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-var', type=int, default=5)
    parser.add_argument('--n-obj', type=int, default=2)
    parser.add_argument('--n-iter', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--problem-name', type=str, default='zdt1')
    parser.add_argument('--decompose', type=str, default='hv1')
    parser.add_argument('--use-plot', type=str, default='Y')
    parser.add_argument('--optimizer-name', type=str, default='SGD' )

    args = parser.parse_args()
    problem_name = args.problem_name
    folder_prefix = os.path.join(os.getcwd(), 'output', problem_name, 'seed_{}'.format(args.seed))
    os.makedirs(folder_prefix, exist_ok=True)
    args.folder_prefix = folder_prefix

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.problem_name == 'lqr2':
        args.n_var = 2
    elif args.problem_name == 'lqr3':
        args.n_var = 3


    if args.problem_name in ['dtlz2', 'lqr3'] or args.problem_name.startswith('RE3'):
        args.n_obj = 3


    if args.decompose == 'hv1':
        lr = hv1_sgd_lr_dict[problem_name]
    elif args.decompose in ['hv2', 'hv2noclip']:
        lr = hv2_sgd_lr_dict[problem_name]
    elif args.decompose == 'mtche':
        lr = mtche_sgd_lr_dict[problem_name]
    elif args.decompose in ['ls', 'tche']:
        lr = tche_sgd_lr_dict[problem_name]
    elif args.decompose == 'epo':
        lr = epo_sgd_lr_dict[problem_name]

    # We now consider 4-obj problems, Date 2023-8-3. 4-obj is done during rebuttal.
    n_iter = args.n_iter
    args.ideal_point = ideal_point_dict[problem_name]
    args.nadir_point = nadir_point_dict[problem_name]

    if args.decompose in ['mtche', 'hv2','epo']:
        pref_eps = 0.05
    else:
        pref_eps = 0.01
    
    pref_eps_max = 1 - pref_eps

    loss_array = [0] * n_iter



    ts = time.time()

    model = main_loop(args=args, nadir_ref=args.nadir_point)

    args.elaps = np.round(time.time() - ts, 2)

    plt.figure()
    plt.plot(loss_array)
    plt.xlabel('Iteration')
    plt.ylabel('PSL loss')
    

    fig_name = os.path.join(args.folder_prefix, 'process_{}_{}.pdf'.format('psl', args.decompose))
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
    print('saved in {}'.format(fig_name))
    plt.close()

    from hvpsl.plotter import plot_main
    plot_main(args, model)
