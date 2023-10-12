from matplotlib import pyplot as plt
import numpy as np
import os
from numpy import array


x = np.linspace(0,1,100)
y = 1 - x**2
plt.plot(x,y, color='tab:blue', label='Pareto front')
radius = 0.02 
cir_x_array = array([0.2,0.4,0.6,0.9])
cir_y_array = 1 - cir_x_array**2
for cur_x, cur_y in zip(cir_x_array, cir_y_array):
    # cur_y = 1-cur_x**2
    circle=plt.Circle((cur_x, cur_y),radius,color='k')
    plt.gca().add_patch(circle)


x = np.linspace(0.2,1.2,100)
y_bound = np.zeros_like(x)
for i in range(len(x)):
    xi = x[i]
    if x[i] <= 0.4:
        y_bound[i] = 1 - cir_x_array[0]**2
    elif xi <= 0.6:
        y_bound[i] = 1 - cir_x_array[1]**2
    elif xi <= 0.9:
        y_bound[i] = 1 - cir_x_array[2]**2
    elif xi <= 1.2:
        y_bound[i] = 1 - cir_x_array[3]**2
        
plt.fill_between(x, y_bound, np.ones_like(x)*1.2, alpha=0.2, color='orange')


circle=plt.Circle((1.2,1.2),0.02,color='k')
plt.gca().add_patch(circle)
plt.text(1.28, 1.2, '$r$', fontsize=16)
plt.text(0.75, 0.75, 'Hypervolume', fontsize=16)

plt.axis('equal')
plt.legend(loc='lower left', fontsize=16)
plt.xlabel('$f_1(x)$', fontsize=14)
plt.ylabel('$f_2(x)$', fontsize=14)


fig_prefix = 'C:\\Users\\xzhang2523\\Desktop\\IJCAI_submit\\HV_PSL\\Figures\\illus'
os.makedirs(name=fig_prefix, exist_ok=True)

fig_name = os.path.join(fig_prefix, 'deng.pdf')
plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
print('saved in:{}'.format(fig_name))
    
plt.show()