import numpy as np
from pymoo.indicators.hv import Hypervolume
from matplotlib import pyplot as plt

ref_point = np.array([1.0, 1.0])
hv_indicator = Hypervolume(ref_point=ref_point)



def solve(a,b,c):
    return (-b - np.sqrt(b**2-4*a*c))/(2*a)


def cal_hv_via_mc(num, pf):
    rho_array = [0]*num
    for i in range(num):
        theta = np.random.random()*np.pi/2
        pref = np.array([np.cos(theta), np.sin(theta)])
        # res = np.c_[pf[:,0] / pref[0], pf[:,1] / pref[1]]
        x = solve(pref[1]**2, -(2*pref[0]*pref[1]+pref[0]**2), pref[0]**2)
        rho = np.linalg.norm([x,1-np.sqrt(x)])
        # rho = np.min(np.max(res, axis=1))
        rho_array[i] = rho**2
    return np.pi/4*np.mean(rho_array)




if __name__ == '__main__':

    candidate = range(1, 100)
    hv_array = np.zeros(len(candidate))
    for idx, num in enumerate(candidate):
        pf_x = np.linspace(0,1,num)
        pf_y = 1-np.sqrt(pf_x)
        pf = np.c_[pf_x, pf_y]
        pf_real = hv_indicator.do(pf)
        hv_polar = 1 - cal_hv_via_mc(num, pf)
        hv_err = pf_real - hv_polar
        hv_array[idx] = hv_err


    plt.plot(candidate, hv_array)
    plt.show()





