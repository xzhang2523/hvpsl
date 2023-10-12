import numpy as np


def get_sphere_surface():
    u, v = np.mgrid[0:np.pi/2:50j, 0:np.pi/2:50j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    return x,y,z
        # ax.plot_surface(x, y, z, color="g", alpha=0.2, label='True')





if __name__ == '__main__':
    print()
    