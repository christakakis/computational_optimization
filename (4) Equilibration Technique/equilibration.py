#Panagiotis Christakakis
#Equilibration Technique

import numpy as np
import pandas as pd

#Initialize A, b, c for testing the algorithm
#'''
A = ([[   100,  -50,  50,    20],
       [   10, 5000,  25, -2000],
       [  200, -100, 100,   -50],
       [ 1000,  250, -50,  -100]])

b = [100, 200, 300, 400]

c = [50, -100, 50, -25]

A = np.float_(A)
b = np.float_(b)
c = np.float_(c)
#'''

#No need to check for columns with all-zero elements because matrix A
#will have already pass through a Presolve Method and these kind of
# rows and columns will have been eliminated
colmax = abs(A).max(axis=0)
colmulti = 1 / abs(A).max(axis=0)

#Initialize rowmulti
rowmulti = [0] * A.shape[0]

#Find indeces of max values (+ duplicate ones)
rowIndex = np.where(abs(A) == colmax)[0]

#Make changes in A, c
for j in range(A.shape[1]):
        A[:,j] = A[:,j] * colmulti[j]
        c[j] = c[j] * colmulti[j]

#Find max of each row (to check if it's "=1" later)
rowmax = abs(A).max(axis=1)
        
#Create a list containing only the rows to parse
all_rows = list(range(len(A[0])))
index_to_iterate = []
for i in all_rows:
    if i not in rowIndex:
        index_to_iterate.append(i)

#Parse the final rows and make the changes that need to be made in A, b
for i in index_to_iterate:
    if rowmax[i] != 1:
        rowmulti[i] = 1 / rowmax[i]
        A[i,:] = A[i,:] * rowmulti[i]
        b[i] = b[i] * rowmulti[i]

#Print the final results after changes
print("Table A: \n", A, "\n")
print("b: ", b, "\n")
print("c: ", c, "\n")
print("RowIndex: ",rowIndex)
print("ColMulti: ",colmulti)
print("RowMulti: ", rowmulti)