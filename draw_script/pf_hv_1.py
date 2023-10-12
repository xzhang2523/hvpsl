from matplotlib import pyplot as plt
import numpy as np
import os





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



    plt.plot(x_pf, y_pf, color='tab:blue', label='Pareto front', linewidth=2)
    plt.fill_between(x, y, np.ones_like(x)*1.2, alpha=0.2, color='orange')
    x2 = np.linspace(0,1,100)
    y2 = 1-x2**2
    circle=plt.Circle((1.2,1.2),0.02,color='k')
    plt.gca().add_patch(circle)
    plt.text(1.28, 1.2, '$r$', fontsize=16)
    plt.text(0.75, 0.75, 'Hypervolume', fontsize=16)



    plt.axis('equal')
    plt.legend(loc='lower left', fontsize=16)
    plt.xlabel('$f_1(x)$', fontsize=14)
    plt.ylabel('$f_2(x)$', fontsize=14)




    # fig_prefix = 'C:\\Users\\xzhang2523\\Desktop\\IJCAI_submit\\HV_PSL\\Figures\\illus'

    fig_prefix = 'C:\\Users\\xzhang2523\\Desktop\\NeurIPS2023Rebuttal\\Figure'
    os.makedirs(name=fig_prefix, exist_ok=True)

    fig_name = os.path.join(fig_prefix, 'ours.pdf')
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
    print('saved in:{}'.format(fig_name))
    plt.show()