import pickle
import os
from matplotlib import pyplot as plt


problem_name = 'RE37'
file_name = os.path.join('C:\\Users\\xzhang2523\\Desktop\\IJCAI_submit\\HV_PSL\\Figures\\exp2', problem_name)


n_iter_array = [100,200,300,500]
# n_iter_array = [200,]





for n_iter in n_iter_array:
    pickle_name = os.path.join(file_name, 'mid_{}.pickle'.format(n_iter))
    with open(pickle_name, 'rb') as f:
        X, J = pickle.load(f)
        
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(J[:,0], J[:,1], J[:,2], color = "green", label='PSL')
        ax.set_xlabel('$f_1(x)$', fontsize=16)
        ax.set_ylabel('$f_2(x)$', fontsize=16)
        ax.set_zlabel('$f_3(x)$', fontsize=16)
        fig_name = os.path.join(file_name, 'mid_pf_{}.pdf'.format(n_iter))
        print('saved in {}'.format(fig_name))
        plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
        
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(X[:,0], X[:,1], X[:,2], color = "green", label='PSL')
        ax.set_xlabel('$x_1$', fontsize=16)
        ax.set_ylabel('$x_2$', fontsize=16)
        ax.set_zlabel('$x_3$', fontsize=16)
        fig_name = os.path.join(file_name, 'mid_ps_{}.pdf'.format(n_iter))
        print('saved in {}'.format(fig_name))
        plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
        
        
    # print()
