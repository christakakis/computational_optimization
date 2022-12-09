#Panagiotis Christakakis
#Eliminate k-ton Equality Constraints

import numpy as np
import pandas as pd

#Initialize A, b, c for testing the algorithm
#'''
A = ([[ 0,  3,  0,  0],
      [ 4, -3,  8, -1],
      [-3,  2,  0,  -4],
      [ 4,  0, -1,  0]])

b = [6, 20, -8, 18]

c = [-2, 4, -2, 2]

Eqin = [0, 0, 0, 0]

A = np.float_(A)
b = np.float_(b)
c = np.float_(c)
#'''

#Initialize variables
k = 2
ind_row = 0
ind_col = 0
c_zero = 0

#Starting with k-ton algorithm
while k != 1:
    for i in range(A.shape[0]-1, -1, -1):
        first_nz = False
        if (A.shape[0] - (np.count_nonzero(A[i], axis=0))) == k:
            for j in range(A.shape[1]-1, -1, -1):
                if A[i][j] != 0 and first_nz == False:
                    b[i] = b[i] / A[i][j]
                    A[i] = A[i] / A[i][j]
                    Eqin[i] = -1
                    ind_col = j
                    ind_row = i
                    first_nz = True            
        for j in range(A.shape[1]-1, -1, -1):
            if A[i][j] != 0 and j == ind_col and i != ind_row:
                b[i] = b[i] - (A[i][j] * b[ind_row])
                A[i] = A[i] - (A[i][j] * A[ind_row])
                if c[ind_col] != 0:
                    c_zero = c_zero + c[ind_col] * b[ind_row]
                    c = c - (c[ind_col] * A[ind_row].transpose())
                    A = np.delete(A, ind_col, axis = 1)
                    c = np.delete(c, ind_col)
    k = k - 1


singleton_ind_col = 0
singleton_ind_row = 0

#Singleton Algorithm
if k == 1:
    x_k = 0
    for i in range(A.shape[0]):
        if np.count_nonzero(A[i], axis=0) == k and Eqin[i] == 0:
            for j in range(A.shape[1]):
                if A[i][j] != 0:
                    singleton_ind_col = j
                    singleton_ind_row = i
                    x_k = b[i] / A[i][j]
                    if x_k != 0:
                        b = b - (x_k * A[:,j])
                        if c[j] != 0:
                            c_zero = c_zero + (c[j] * x_k)
                            A = np.delete(A, singleton_ind_row, axis = 0)
                            c = np.delete(c, singleton_ind_col)
                            b = np.delete(b, singleton_ind_row)
                            Eqin = np.delete(Eqin, singleton_ind_col)
                            break
                            
        break

#Print the final results after changes
print("Table A: \n", A, "\n")                
print("b: \n", b, "\n")
print("c: \n", c, "\n")
print("C0: ", c_zero, "\n")
print("Eqin: \n", Eqin, "\n")