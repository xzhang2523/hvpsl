from torch import Tensor
from numpy import array
import numpy as np

def mgda_direction(g1, g2):
      alpha = ((g2-g1)@g2) / ((g2-g1)@(g2-g1))
      alpha = np.clip(alpha,0,1)
      return alpha, 1-alpha
    
    
    
    
if __name__ == '__main__':
      a = array([1,2,3])
      b = array([1,2,4])
      r,r2 = mgda_direction(a,b)
      print(r,r2)
      
      
  