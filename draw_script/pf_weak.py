from matplotlib import pyplot as plt
import numpy as np
import os

'''
    Figures\pf_weak.py
    This is use to draw Figure 1 in the paper ((weakly)Pareto dominance).
'''



if __name__ == '__main__':
    x = np.linspace(0,1.2,100)
    y = np.zeros_like(x)
    for i in range(len(x)):
        if x[i]<1:
            y[i] = 1 - x[i]**2
        else:
            y[i] = 0
    x_pf = np.linspace(0, 1, 100)
    y_pf = 1 - x_pf**2
    plt.plot(x_pf, y_pf, color='tab:blue', label='Pareto front')
    plt.fill_between(x, y, np.ones_like(x)*1.2, alpha=0.2, color='orange')

    # plot weak
    plt.scatter(0, 1.1, color='y')
    plt.scatter(1.1, 0, color='y', label='Weakly Pareto solutions')


    plt.scatter(0.4, 1-0.4**2, color='tab:blue')
    plt.scatter(0.8, 1-0.8**2, color='tab:blue')
    plt.scatter(0.6, 1-0.6**2, color='tab:blue', label='Pareto solutions')


    plt.scatter(0.7, 1.0, color='tab:red')
    plt.scatter(1.0, 0.7, color='tab:red', label='Dominated solutions')
    plt.scatter(1.2,1.2, color='k')
    plt.text(1.2+0.05, 1.2, '$r$', fontsize=16)
    plt.text(1.0, 0.4, 'Hypervolume', fontsize=16)

    plt.axis('equal')
    plt.legend(loc='lower left', fontsize=12)
    plt.xlabel('$f_1(x)$', fontsize=14)
    plt.ylabel('$f_2(x)$', fontsize=14)


    fig_prefix = 'C:\\Users\\xzhang2523\\Desktop\\IJCAI_submit\\HV_PSL\\Figures\\illus'
    os.makedirs(name=fig_prefix, exist_ok=True)

    fig_name = os.path.join(fig_prefix, 'weak.pdf')
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
    print('saved in:{}'.format(fig_name))
    plt.show()