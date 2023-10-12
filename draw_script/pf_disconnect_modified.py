from matplotlib import pyplot as plt
import os
from time import time
import numpy as np


'''
    This is for the rebuttal figure
'''



if __name__ == '__main__':
    point_num = 1000
    x_pf = np.c_[np.linspace(0, 0.2, point_num), np.linspace(0.6, 1.0, point_num)]
    y_pf = 1 - np.sqrt(x_pf)

    x = np.linspace(0, 1.0, point_num)
    y = np.zeros_like(x)
    # y = 1 - np.sqrt(x)

    y1 = 1 - np.sqrt(0.2)
    for i in range(point_num):
        if x[i] > 0.2 and x[i] < 0.6:
            y[i]=y1
        else:
            y[i] = 1-np.sqrt(x[i])


    xx = np.linspace(0.2, 1.0, 100)
    yy = 0.6/0.8 * (xx-1) + 1
    plt.fill_between(xx, np.ones_like(yy)*0.4, yy, color='g', alpha=0.2)

    yy2 = 0.6/0.8 * (xx-1) + 1
    for idx, (xi, yi) in enumerate(zip(xx, yy2)):
        if xi < 0.4:
            yy2[idx] = 1-np.sqrt(0.2)
        
    plt.fill_between(xx, yy2, np.ones_like(yy)*1, color='r', alpha=0.1)
        

    # plt.scatter([1.0,], [1.0,])
    plt.scatter([0.2,], [1.0-np.sqrt(0.2),], color='k', label='Solution',s=60)

    # plt.plot([0.2,1.0],[1.0,1.0], color='g')
    plt.plot([0.2,0.2],[1.0-np.sqrt(0.2),1.0], color='r')
    plt.plot([0.4,1],[1.0-np.sqrt(0.2),1.0], '--',color='r', linewidth=3)

    plt.plot([0.2, 1.0], [0.4, 0.4],'-', color='g', linewidth=1)
    
    plt.text(0.9, 0.85,'$\\theta$',fontsize=18)
    
    plt.text(0.6, 0.67,'$\\rho(\\theta)$',fontsize=18)
    plt.text(0.55, 0.38,'$r_1-f_1$',fontsize=14)
    plt.text(0.22, 0.76,'$r_2-f_2$',fontsize=14)
    
    plt.text(1.01, 1.01, '$r$', fontsize=14)

    plt.axis('equal')
    plt.plot(x, y, label='Attainment surface', color='blue', linewidth=1)
    
    plt.plot(x_pf, y_pf, linewidth=2, color='blue')
    
    plt.xlabel('$f_1(x)$',fontsize=16)
    plt.ylabel('$f_2(x)$',fontsize=16)

    plt.legend(fontsize=14)
    fig_name = os.path.join('C:\\Users\\xzhang2523\\Desktop\\IJCAI_submit\\IJCAI rebuttal', 'disconnect.pdf') 
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()
    print('saved in {}'.format(fig_name))