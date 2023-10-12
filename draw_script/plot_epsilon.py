import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(0, 10, 1000)
y = sigmoid(x)

plt.plot(x, y, label='Estimated HV')
plt.plot(x, 1.05*np.ones_like(x), '--', label='True HV')
# plt.title('Sigmoid Function')
plt.xlabel('Iteration', fontsize=18)
plt.ylabel('$\mathcal{H}_r$', fontsize=18)
plt.text(8,1.02,'$\epsilon$', fontsize=18)
plt.legend(fontsize=18)
plt.show()


