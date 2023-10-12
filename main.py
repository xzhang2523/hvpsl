import numpy as np

from matplotlib import pyplot as plt


rho1,rho2 = 2,2


t = np.linspace(0,1,100)
theta1 = -1/(1+np.exp(-2.18 - rho1*t**2+(3.34+rho1)*t))
theta2 = -1/(1+np.exp(1.15 - rho2*t**2+(-3.34+rho1)*t))



plt.plot(t,theta1)
plt.plot(t,theta2)

plt.show()
