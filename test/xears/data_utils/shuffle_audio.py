"""Utilities audio convert."""
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

#given train_ori A and B
#return train_set by shuffle
def shuffle_two_audio(A, B):
   print("shuffle_two_audio")
   print("A.shape"+str(A.shape))
   a_c = A.shape[0]#channel
   a_u = A.shape[1]#unit
   a_s = A.shape[2]#step

   print("B.shape"+str(B.shape))
   b_c = B.shape[0]#channel
   b_u = B.shape[1]#unit
   b_s = B.shape[2]#step

   #np.concatenate((first, second), axis=1)
   train_set_x = np.zeros((2,2))
   train_set_y = np.zeros((1,2))
   return train_set_x, train_set_y






