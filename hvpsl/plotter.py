from matplotlib import pyplot as plt
import os

# torch related
import torch
from torch import Tensor

# self-lib
from hvpsl.psl_util import objective, uniform_sphere_pref
from hvpsl.pf_util import load_real_pf, load_re_pf_norm

from hvpsl.indicators import get_ind_sparsity, get_ind_range, get_ind_hv




# 3rd lib
import csv
from numpy import array
import numpy as np





def plot_main(args, model):

    if args.n_obj == 2:
        pref = Tensor(uniform_sphere_pref(m=args.n_obj, n=60))
    else:
        pref = Tensor(uniform_sphere_pref(m=args.n_obj, n=80))

    with torch.no_grad():
        if args.n_obj == 3:
            x = model(pref)
            J = objective(args, x).numpy()
            ax = plt.axes(projection='3d')
            ax.view_init(elev=46, azim=33)
            ax.scatter3D(J[:, 0], J[:, 1], J[:, 2], color="green", label='PSL')
            ax.set_xlabel('$f_1(x)$', fontsize=16)
            ax.set_ylabel('$f_2(x)$', fontsize=16)
            ax.set_zlabel('$f_3(x)$', fontsize=16)
            fig_name = os.path.join(args.folder_prefix, '{}_{}.pdf'.format('psl', args.decompose))
            plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
            print('saved in {}'.format(fig_name))

        elif args.n_obj == 2:
            plt.figure()
            x = model(pref)
            pref_np = pref.numpy()
            if args.problem_name in ['zdt1', 'zdt2', 'vlmop2'] or args.problem_name.startswith('RE'):
                pref_np = pref_np * 0.4

            # if args.problem_name.startswith('RE'):
            #     J = problem.evaluate(x).numpy()
            # else:
            #     J = loss_function(x, problem=args.problem_name).numpy()
            J = objective(args, x).numpy()


            if args.problem_name.startswith('RE'):
                pf_real = load_re_pf_norm( problem_name=args.problem_name )
            else:
                pf_real = load_real_pf( problem_name=args.problem_name )

            for prefi, ji in zip(pref_np, J):
                if args.decompose == 'hv1':
                    nadir_np = args.nadir_point.numpy()
                    plt.plot([nadir_np[0] - prefi[0], ji[0]], [nadir_np[1] - prefi[1], ji[1]], color='tomato')
                else:
                    plt.plot([prefi[0], ji[0]], [prefi[1], ji[1]], color='tomato')

            plt.scatter(J[:, 0], J[:, 1], label='PSL', color='gold')
            plt.plot(pf_real[:, 0], pf_real[:, 1], label='True', color='k')
            if args.decompose == 'hv1':
                plt.plot(nadir_np[0] - pref_np[:, 0], nadir_np[1] - pref_np[:, 1], color='skyblue', label='Pref.')
            else:
                plt.plot(pref_np[:, 0], pref_np[:, 1], color='skyblue', label='Pref.')
            plt.axis('equal')
            plt.legend()
            plt.xlabel('$f_1$', fontsize=16)
            plt.ylabel('$f_2$', fontsize=16)

            fig_name = os.path.join(args.folder_prefix, '{}_{}.pdf'.format('psl', args.decompose))

            if args.use_plot == 'Y':
                plt.show()

            plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
            print('fig saved in :{}'.format(fig_name))
            plt.close()




        if args.n_obj == 2:
            J_theta = array([np.pi / 2 - np.arctan2(*elem) for elem in J])
            range_val = np.round(np.max(J_theta) - np.min(J_theta), 2)
        else:
            range_val = np.round(get_ind_range(J), 2)

        hv_indicator = get_ind_hv(args)
        hv_val = np.round(hv_indicator.do(J), 2)
        if args.n_obj == 2:
            sparsity_val = get_ind_sparsity(J) * 1e3
        else:
            sparsity_val = get_ind_sparsity(J) * 1e7


        print('PSL True hv:{:.2f}'.format(hv_val))
        csv_file_name = os.path.join(args.folder_prefix, '{}.csv'.format(args.decompose))
        with open(csv_file_name, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(['HV', 'Range', 'sparsity_val', 'Time'])
            spamwriter.writerow([str(hv_val), str(range_val), str(np.round(sparsity_val, 3)), args.elaps])
        print('csv saved in {}'.format(csv_file_name))
