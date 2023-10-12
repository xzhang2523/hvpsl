import numpy as np


def solve(a,b,c):
    return (-b+np.sqrt(b**2-4*a*c))/(2*a)


print(solve(10,-14.4,9*0.64-1))