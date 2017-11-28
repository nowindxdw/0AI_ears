"""Utilities audio convert."""
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

#given train_ori A and B
#return train_set by shuffle
def shuffle_two_audio(A, B):
   #print("shuffle_two_audio")
   #print("A.shape"+str(A.shape))
   a_c = A.shape[0]#channel
   a_u = A.shape[1]#unit
   a_s = A.shape[2]#step

   #print("B.shape"+str(B.shape))
   b_c = B.shape[0]#channel
   b_u = B.shape[1]#unit
   b_s = B.shape[2]#step

   #mix A and B to C
   c_c = a_c
   c_u = min(a_u,b_u)
   c_s = a_s

   C = np.zeros((c_c,c_u,c_s))
   #print("C.shape"+str(C.shape))
   for i in range(c_u):
      a_temp = np.random.randint(0,a_u)
      b_temp = np.random.randint(0,b_u)
      #print("a_temp"+str(a_temp))
      #print("b_temp"+str(b_temp))
      #print(A[:, a_temp, 0:int(a_s/2)].shape)
      #print(B[:, b_temp, 0:int(b_s/2)].shape)
      C[:,i,:] =  np.concatenate((A[:, a_temp, 0:int(a_s/2)], B[:, b_temp, 0:int(b_s/2)]), axis = -1)
      #print(C[:,i,:].shape)
   #print("C",C)
   train_set_x = np.concatenate((A,B),axis=1)
   train_set_x = np.concatenate((train_set_x,C),axis=1)
   train_u = a_u+b_u+c_u
   #dtype default float64, but label must be integer
   train_set_y = np.zeros(train_u,dtype=np.int)
   train_set_y[a_u] = 1
   train_set_y[a_u:b_u+a_u] = 2
   return train_set_x, train_set_y






