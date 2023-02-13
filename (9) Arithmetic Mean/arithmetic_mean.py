# Panagiotis Christakakis
# Arithmetic Mean - Scaling Technique

# Import Libraries
import numpy as np
import pandas as pd
import scipy

# Remove scientific notation (i.e. numbers with e)
np.set_printoptions(suppress=True)

# Define rows, cols lenght
rows = len(A)
cols = len(A[1])

# Initialize original r, s and their non_zeros
r = [0] * rows
s = [0] * cols

r_non_zeros = [0] * rows
s_non_zeros = [0] * cols

# Create r
for i in range(rows):
    row_non_zeros = 0
    for j in range(cols):
        if A[i][j] != 0:
            row_non_zeros += 1
            r_non_zeros[i] = row_non_zeros
            r[i] += abs(A[i][j]) 

print("Table A has the following Non-Zero elements per ROW:", r_non_zeros)
            
for i in range(rows):
    r[i] = r_non_zeros[i] / r[i]
            
# Update A and b
for i in range(rows):
    b[i] = np.dot(b[i], r[i])
    for j in range(cols):
        A[i][j] = np.dot(A[i][j], r[i])
        
# Create s
for j in range(cols):
    col_non_zeros = 0
    for i in range(rows):
        if A[i][j] != 0:
            col_non_zeros += 1
            s_non_zeros[j] = col_non_zeros
            s[j] += abs(A[i][j])
            
print("Table A after the update has the following Non-Zero elements per COL:", s_non_zeros)

for j in range(cols):
    s[j] = s_non_zeros[j] / s[j]
    
# Update A and c
for j in range(cols):
    c[j] = np.dot(c[j], s[j])
    for i in range(rows):
        A[i][j] = np.dot(A[i][j], s[j])
        
# Convert r, s to numpy arrays for rounding
r = np.asarray(r)
s = np.asarray(s)

# Rounding for simplicity
A = A.round(decimals=5)
b = b.round(decimals=5)
c = c.round(decimals=5)
r = r.round(decimals=5)
s = s.round(decimals=5)

print("\nA: \n", A)
print("\nb: ", b)
print("\nc: ", c)
print("\nr: ", r)
print("\ns: ", s)
