import numpy as np
from numpy import array
import pickle
from matplotlib import pyplot as plt


from pymoo.indicators.hv import Hypervolume
ref_point = np.array([1.2, 1.2])
hv_indicator = Hypervolume(ref_point=ref_point)
import sys
sys.path.append('D:\code\Paper_IJCAI\manifold_rl')
import os
from reproblem import load_re_pf, load_re_pf_norm


problem_name = 'RE24'
n_iter = 100
file_name = 'D:\\code\\Paper_IJCAI\\draw_script\\data\\{}_{}.pickle'.format(problem_name, n_iter)
with open(file_name, 'rb') as f:
    (pref_np, X_np, J_np) = pickle.load(f)
    true_pf = load_re_pf_norm(problem_name)
    sort_idx = np.argsort(true_pf[:,0])
    true_pf = true_pf[sort_idx]
    
    hv_val = hv_indicator.do(J_np)
    hv_val_true = hv_indicator.do(true_pf)
    print('hv_val:\t{:.2f}'.format(hv_val))
    print('hv_val true:\t{:.2f}'.format(hv_val_true))
    
    
    down_sample = 10
    # pref_np = pref_np[::down_sample]
    X_np = X_np[::down_sample]
    J_np = J_np[::down_sample]
    theta = np.linspace(0, np.pi/2, len(X_np))
    pref_np = np.c_[np.sin(theta), np.cos(theta)]
    
    plt.plot(J_np[:,0], J_np[:,1],color='gold', label='Solution $f(x_\\beta(\lambda))$', linewidth=4)
    
    J_max = np.max(J_np, axis=0)
    J_min = np.min(J_np, axis=0)
    J_np_hv = np.concatenate((J_np, array([[1.0,J_min[1]]])))
    plt.fill_between(J_np_hv[:,0], J_np_hv[:,1], np.ones_like(J_np_hv[:,0]) * 1.0 ,alpha=0.2, color='gold', label='HV of PSL')
    
    circle=plt.Circle((1.0,1.0),0.02,color='k')
    plt.gca().add_patch(circle)
    plt.text(1.08, 1.0, 'ref. point', fontsize=16)

    
    plt.scatter(true_pf[:,0], true_pf[:,1], color='tab:blue', label='MOEAD solutions', s=1.5)
    
    
    true_J_min = np.min(true_pf, axis=0)
    true_J_max = np.max(true_pf, axis=0)
    
    true_pf_hv = np.concatenate((true_pf, array([[1.0,true_J_min[1]]])))
    # plt.fill_between(true_pf_hv[:,0], true_pf_hv[:,1], np.ones_like(true_pf_hv[:,0]) * 1.0 ,alpha=0.2, color='tab:blue', label='HV of PSL')
    
    
    
    
    
    
    plt.legend(loc='lower right')
    plt.axis('equal')
    plt.xlabel('$f_1(x)$')
    plt.ylabel('$f_2(x)$')
    fig_prefix = 'C:\\Users\\xzhang2523\\Desktop\\IJCAI_submit\\HV_PSL\\Figures\\illus'
    fig_name = os.path.join(fig_prefix, '{}_{}.pdf'.format(problem_name, n_iter))
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
    print('saved in:{}'.format(fig_name))



    plt.figure()
    # ax = plt.axes(projection ="3d")
    
    down_sample_2 = 2
    pref_np_down = pref_np[::down_sample_2]
    X_np_down = X_np[::down_sample_2]
    for prefi, xi in zip(pref_np_down, X_np_down):
        plt.plot([prefi[0], xi[0]], [prefi[1], xi[1]], color='tomato')
    
    plt.plot(X_np_down[:,0], X_np_down[:,1], color='gold', label='Solution $f(x_\\beta(\lambda))$', linewidth=4)
    plt.plot(pref_np_down[:,0], pref_np_down[:,1], color='skyblue', label='Preference $\lambda$', linewidth=4)
    
    plt.scatter(X_np_down[:,0], X_np_down[:,1], color='gold',s=1)
    plt.scatter(pref_np_down[:,0], pref_np_down[:,1], color='skyblue',s=1)
    plt.legend(loc='lower right')
    plt.axis('equal')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    
    fig_name = os.path.join(fig_prefix, 'PS_{}_{}.pdf'.format(problem_name, n_iter))
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
    print('saved in:{}'.format(fig_name))
    plt.show()
    