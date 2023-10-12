import numpy as np
from matplotlib import pyplot as plt
import os



def vlmop1(x):
    f1 = 1 - np.exp( -(x-1)**2 )
    f2 = 1.2 * ( 1 - np.exp( -(x+1)**2 ))
    return np.column_stack([f1, f2])


if __name__ == '__main__':
    
    FONT_SIZE = 16
    x = np.linspace(-2, 2, 100)
    y = vlmop1(x)
    
    ls = 0.5*(y[:,0] + y[:,1])
    tche = np.max(y, axis=1)
    
    plt.plot(x, y[:,0], label='$f_1(x)$', linewidth=2, color='k')
    plt.plot(x, y[:,1], label='$f_2(x)$', linewidth=2, color='k')
    
    plt.plot(x, ls, label='$LS$', linewidth=2, color='b')
    plt.plot(x, tche, label='$\\rho(\\theta)$', linewidth=3, color='r')
    
    
    
    plt.text(-1.1, 0.2, '$f_1(x)$', fontsize=FONT_SIZE, color='k')
    plt.text(0.8, 0.2, '$f_2(x)$', fontsize=FONT_SIZE, color='k')
    
    plt.text(-0.6, 0.5, 'LS', fontsize=FONT_SIZE, color='b')
    plt.text(0.2, 0.8, '$\\rho(\\theta)$', fontsize=FONT_SIZE, color='r')
    
    
    plt.xlabel('$x$', fontsize=FONT_SIZE)
    plt.ylabel('$y$', fontsize=FONT_SIZE)
    
    
    fig_prefix = 'C:\\Users\\xzhang2523\\Desktop\\IJCAI_submit\\HV_PSL\\Figures\\illus'
    os.makedirs(name=fig_prefix, exist_ok=True)
    
    
    
    fig_name = os.path.join(fig_prefix, 'quasi.pdf')
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
    print('fig saved in :{}'.format(fig_name) )
    
    
    # plt.legend(fontsize = FONT_SIZE)
    
    plt.show()