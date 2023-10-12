from matplotlib import pyplot as plt
import numpy as np
import os

'''
    Figures\pf_hv_proof.py
    Used as Figure 3(a).
'''


def solve(a,b,c):
    return (-b+np.sqrt(b**2-4*a*c))/(2*a)


if __name__ == '__main__':
    x = np.linspace(0,1.2,100)
    y = np.zeros_like(x)
    for i in range(len(x)):
        if x[i]<1:
            y[i] = 1 - x[i]**2
        else:
            y[i] = 0
            
    
    fig, ax = plt.subplots()
    plt.text(0.2, 1.0, 'Hypervolume', fontsize=14)
    plt.plot(x,y)
    plt.plot([0,0],[1,1.2], color='tab:blue', label='Weakly Pareto front', linewidth=2)
    plt.fill_between(x, y, np.ones_like(x)*1.2, alpha=0.2, color='orange')


    x2 = np.linspace(0, 1, 100)
    y2 = 1-x2**2
    plt.fill_between(x2, np.zeros_like(x2), y2, alpha=0.2, color='tab:blue')


    xp = solve(1,3,-3*0.8-1)
    yp = 1-xp**2
    plt.plot([xp, 1.2], [yp, 1.2], color='k')
    circle=plt.Circle((xp, yp),0.01,color='k')
    plt.gca().add_patch(circle)


    xp2 = 1.1
    yp2 = 3*(xp2-0.8)
    rho_p2 = np.linalg.norm([1.2-xp2, 1.2-yp2])
    circle=plt.Circle((xp2, yp2),0.01,color='k')


    theta = np.arctan2(1.2-xp2, 1.2-yp2)
    theta_arr = np.linspace(np.pi/2*3-theta, np.pi/2*3,20)
    x = rho_p2 * np.cos(theta_arr) + 1.2
    y = rho_p2 * np.sin(theta_arr) + 1.2
    plt.plot(x, y, color='k')
    plt.text(1.24, 0.8, '$\\theta$', fontsize=16)


    circle=plt.Circle((1.2,1.2),0.02,color='k')
    plt.gca().add_patch(circle)
    plt.text(1.28, 1.2, '$r$', fontsize=16)
    
    
    circle=plt.Circle((0.6, 0.8),0.01,color='k')
    plt.gca().add_patch(circle)
    
    plt.text(0.6, 0.82, '$f(x)$', fontsize=16)
    
    
    circle=plt.Circle((1.068, 0.8),0.01,color='k')
    plt.plot([0.6,1.068],[0.8,0.8],color='k', linestyle='--')
    plt.gca().add_patch(circle)
    # plt.text(1.05, 0.8, 'x', fontsize=16)
    
    
    plt.text(1.09, 0.5, '$\\rho(\\theta)$', fontsize=16)
    plt.text(0.80, 1.02, '$\\rho(x, \\theta)$', fontsize=16)
    
    
    # plt.text(0.6, 0.8, 'x', fontsize=16)
    
    

    plt.axis('equal')
    plt.xlabel('$f_1(x)$', fontsize=14)
    plt.ylabel('$f_2(x)$', fontsize=14)



    ax.annotate("",
            xy=(0.96, 0.197), xycoords='data',
            xytext=(1.3, 1.18), textcoords='data',
            arrowprops=dict(arrowstyle="|-|",
                            connectionstyle="arc3"),
            )
    
    ax.annotate("",
            xy=(0.98, 0.82), xycoords='data',
            xytext=(1.13, 1.24), textcoords='data',
            arrowprops=dict(arrowstyle="|-|",
                            connectionstyle="arc3"),
            )
    
    
    
    plt.legend(fontsize=16)
    fig_prefix = 'C:\\Users\\xzhang2523\\Desktop\\IJCAI_submit\\HV_PSL\\Figures\\illus'
    fig_name = os.path.join(fig_prefix, 'ours_2.pdf')
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
    print('saved in:{}'.format(fig_name))
    plt.show()