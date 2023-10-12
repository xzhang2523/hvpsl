from matplotlib import pyplot as plt
import numpy as np
import pickle



file_name = 'D:\\code\\Paper_IJCAI\\draw_script\\data\\RE21.pickle'
with open(file_name, 'rb') as f:
    (pref_np, X_np, J_np) = pickle.load(f)
    down_sample = 10
    # pref_np = pref_np[::down_sample]
    X_np = X_np[::down_sample]
    J_np = J_np[::down_sample]
    theta = np.linspace(0, np.pi/2, len(X_np))
    
    pref_np = 1.2 - np.c_[np.cos(theta), np.sin(theta)] * 0.4
    
    
    for idx, (prefi, Ji) in enumerate(zip(pref_np, J_np)):
        if idx == 0:
            plt.plot([prefi[0], Ji[0]], [prefi[1], Ji[1]], color='tomato', label='Mapping',linewidth=1)
        else:
            plt.plot([prefi[0], Ji[0]], [prefi[1], Ji[1]], color='tomato', linewidth=1)
    
    plt.plot(J_np[:,0], J_np[:,1],color='gold', label='Solution $f(x_\\beta(\\theta))$', linewidth=4)
    plt.plot(pref_np[:,0], pref_np[:,1],color='skyblue', label='Polar angle $\\Theta$', linewidth=4)
    plt.axis('equal')
    
    
    plt.fill_between(J_np[:,0], J_np[:,1], np.ones_like(J_np[:,0]) * np.max(J_np[:,1]), alpha=0.2, color='gold')
    # nadir = np.max(J_np, axis=0)
    
    plt.plot([0.8,1.2], [1.2,1.2], color='k', linestyle='--', linewidth=1)
    plt.plot([1.2,1.2], [0.8,1.2], color='k', linestyle='--', linewidth=1)
    
    
    nadir = np.array([1.2, 1.2])
    plt.text(nadir[0]+0.02, nadir[1]+0.02, '$r$', fontsize=16)
    circle=plt.Circle(nadir, 0.02, color='k')
    plt.gca().add_patch(circle)
    # plt.text(0.4, 0.6, 'Hypervolume', fontsize=16)
    plt.legend(loc='lower left')
    plt.xlabel('$f_1(x)$', fontsize=14)
    plt.ylabel('$f_2(x)$', fontsize=14)    
    import os
    fig_prefix = 'C:\\Users\\xzhang2523\\Desktop\\IJCAI_submit\\HV_PSL\\Figures\\illus'
    os.makedirs(name=fig_prefix, exist_ok=True)
    
    
    fig_name = os.path.join(fig_prefix, 'first_ours.pdf')
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
    print('fig saved in :{}'.format(fig_name) )
    plt.show()
        
    