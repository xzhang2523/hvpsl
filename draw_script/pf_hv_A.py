from matplotlib import pyplot as plt
import numpy as np
import os

'''
    This is the figure of HV-PSL2, as described in the appendix.
'''





if __name__ == '__main__':


    x = np.linspace(0,1.2,100)
    y = np.zeros_like(x)
    for i in range(len(x)):
        if x[i]<1:
            y[i] = 1 - x[i]**2
        else:
            y[i] = 0
            

    plt.plot(x,y)
    plt.plot([0,0],[1,1.2], color='tab:blue')
    plt.fill_between(x, y, np.ones_like(x)*1.2, alpha=0.2, color='orange')



    x2 = np.linspace(0,1,100)
    y2 = 1-x2**2
    plt.fill_between(x2, np.zeros_like(x2), y2, alpha=0.2, color='tab:blue')

    plt.text(0.5, 0.4, 'A', fontsize=16)
    plt.text(0.8, 0.8, 'Hypervolume', fontsize=16)


    plt.plot([0, 0.4], [0, 1-0.4**2], color='tab:blue')
    circle=plt.Circle((0,0),0.01,color='k')
    plt.gca().add_patch(circle)
    circle=plt.Circle((0.4,1-0.4**2),0.01,color='k')
    y3=1-0.4**2
    x3=0.4

    plt.text(0.25, 0.6, '$\overline{\\rho}(\\theta)$', fontsize=16)

    theta = np.arctan2(y3, x3)
    theta_arr = np.linspace(0, theta,20)
    x = 0.15 * np.cos(theta_arr)
    y = 0.15 * np.sin(theta_arr)
    plt.plot(x, y, color='tab:blue')
    plt.text(0.15, 0.15, '$\\theta$', fontsize=16)

    plt.gca().add_patch(circle)

    circle=plt.Circle((1.2,1.2),0.02,color='k')
    plt.gca().add_patch(circle)
    plt.text(1.28, 1.2, '$r$', fontsize=16)
    plt.text(0.05, -0.02, 'Ideal point', fontsize=16)

    plt.axis('equal')
    plt.xlabel('$f_1(x)$', fontsize=16)
    plt.ylabel('$f_2(x)$', fontsize=16)

    fig_prefix = 'C:\\Users\\xzhang2523\\Desktop\\IJCAI_submit\\HV_PSL\\Figures\\illus'
    fig_name = os.path.join(fig_prefix, 'ours_psl_2.pdf')
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
    print('saved in:{}'.format(fig_name))
    plt.show()