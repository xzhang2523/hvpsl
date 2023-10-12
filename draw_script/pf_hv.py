from matplotlib import pyplot as plt
import numpy as np
import os


def solve(a,b,c):
    return (-b+np.sqrt(b**2-4*a*c))/(2*a)


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
# plt.fill_between(x2, np.zeros_like(x2), y2, alpha=0.2, color='tab:blue')




xp = solve(1,3,-3*0.8-1)
yp = 1-xp**2

plt.plot([xp, 1.2], [yp, 1.2], color='tab:blue')



circle=plt.Circle((xp, yp),0.01,color='k')
plt.gca().add_patch(circle)

xp2 = 1.1
yp2 = 3*(xp2-0.8)
rho_p2 = np.linalg.norm([1.2-xp2, 1.2-yp2])
circle=plt.Circle((xp2, yp2),0.01,color='k')
# plt.gca().add_patch(circle)

theta = np.arctan2(1.2-xp2, 1.2-yp2)


plt.text(0.88, 0.6, '$\\rho(\\theta)$', fontsize=16)
plt.text(0.4, 1.0, 'Hypervolume', fontsize=16)


theta_arr = np.linspace(np.pi/2*3-theta, np.pi/2*3,20)
x = rho_p2 * np.cos(theta_arr) + 1.2
y = rho_p2 * np.sin(theta_arr) + 1.2
plt.plot(x, y, color='tab:blue')
plt.text(1.24, 0.8, '$\\theta$', fontsize=16)


circle=plt.Circle((1.2,1.2),0.02,color='k')
plt.gca().add_patch(circle)
plt.text(1.28, 1.2, '$r$', fontsize=16)



plt.axis('equal')
plt.xlabel('$f_1(x)$', fontsize=14)
plt.ylabel('$f_2(x)$', fontsize=14)


fig_prefix = 'C:\\Users\\xzhang2523\\Desktop\\IJCAI_submit\\HV_PSL\\Figures\\illus'
os.makedirs(name=fig_prefix, exist_ok=True)

fig_name = os.path.join(fig_prefix, 'ours_2.pdf')
plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
print('saved in:{}'.format(fig_name))
plt.show()