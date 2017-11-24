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
   c_u = a_u
   c_s = min(a_s,b_s)

   C = np.zeros((c_c,c_u,c_s))
   #print("C.shape"+str(C.shape))
   for i in range(c_s):
      a_temp = np.random.randint(0,a_s)
      b_temp = np.random.randint(0,b_s)
      #print("a_temp"+str(a_temp))
      #print("b_temp"+str(b_temp))
      #print(A[:, 0:int(a_u/2), a_temp])
      #print(B[:, 0:int(a_u/2), b_temp])
      C[:,:,i] =  np.concatenate((A[:, 0:int(a_u/2), a_temp], B[:, 0:int(a_u/2), b_temp]), axis = 1)
      #print(C[:,:,i])
   #print("C",C)
   train_s = a_s+b_s+c_s
   train_set_x = np.concatenate((A,B),axis=-1)
   train_set_x = np.concatenate((train_set_x,C),axis=-1)
   train_set_y = np.zeros((1,train_s))
   train_set_y[0,:a_s] = 1
   train_set_y[0,a_s:b_s+a_s] = -1
   return train_set_x, train_set_y






