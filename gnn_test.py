import numpy as np
from numpy import array



M = array([[0,1/3,1/4,0,0,1/2],
           [1/3,0,0,1/2,1/2,0],
           [1/3,0,0,1/2,1/2,1/2],
           [0,1/3,1/4,0,0,0],
           [0,1/3,1/4,0,0,0],
           [1/3,0,1/4,0,0,0],
           ])

res = M@M@M@M@M@M@M@M@M
print()