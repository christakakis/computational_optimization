#Panagiotis Christakakis
#Compressed Sparse Row (CSR)

import numpy as np
from scipy.sparse import random

#random(m, n, density = d) where m,n are rows and 
#columns of the matrix and d represents density of the matrix
A = random(15, 6, density = 0.18)
A.toarray()
#Prints the random sparse array
print(A.toarray(), "\n")

#Initializing nnZ sum and lists

nz_sum = 0
zero_row = [False] * A.shape[0]
Anz = []
JA = []
IA = [0] * (A.shape[0] + 1)

#Iterating through matrix A
for i in range(A.shape[0]):
    zero_count = 0
    for j in  range(A.shape[1]):
        if A.toarray()[i][j] == 0:
            zero_count = zero_count + 1
        if A.toarray()[i][j] != 0:
            nz_sum = nz_sum + 1
            Anz.append(A.toarray()[i][j])
            JA.append(j+1)
            if IA[i] == 0:
                IA[i] = nz_sum
    if zero_count == (A.shape[1]):
        zero_row.insert(i, True)
    if i == (A.shape[0] - 1):
        IA[A.shape[0]] = nz_sum + 1
        for k in range(len(IA), 0, -1):
            if IA[k-1] == 0:
                IA[k-1] = IA[k]

print("Non-Zero counter: ", nz_sum, "\n")
print("Anz: ", Anz)
print("JA: ", JA)
print("IA: ", IA)