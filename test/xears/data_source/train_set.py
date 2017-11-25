from __future__ import absolute_import
from __future__ import print_function

import numpy as np

#test shuffle data
def train_A_x():
   np.random.seed(1)
   return np.random.rand(2,4,10)#(nc,slen,step)
def train_B_x():
   np.random.seed(3)
   return np.random.rand(2,6,10)#(nc,slen,step)

#A_x = train_A_x()
#print(A_x)
#B_x = train_B_x()
#print(B_x)