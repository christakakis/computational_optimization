#Panagiotis Christakakis
#Exterior Point Algorithm

#Import Libraries
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import math

#Initialize A, b, c for testing the algorithm
A = ([[  1,  2,  1,  3],
      [ -4,  1, -2,  1],
      [  3, -2, -1,  2]])

b = [12, -4, 8]

c = [-2, 1, -3, -1]

Eqin = [-1, 1, -1]

A = np.float_(A)
b = np.float_(b)
c = np.float_(c)

#Define m, n lenght
m = len(A)
n = len(A[1])

#Initialize some array lists
B = np.array([])
N = np.array([])
P = np.array([])
Q = np.array([])
L = np.array([])

ineq = 0
for i in range (0,m):
    if Eqin[i] != 0:
        ineq = ineq + 1
print(ineq)

to_concut = np.zeros((m, ineq))
to_concut

add_to_col = 0
for i in range(len(to_concut)):
    if Eqin[i] == -1:
        to_concut[i,add_to_col] = 1
        add_to_col = add_to_col + 1
    if Eqin[i] == 1:
        to_concut[i,add_to_col] = -1
        add_to_col = add_to_col + 1
to_concut

A_new = np.concatenate((A,to_concut), axis=1)
print(A_new)

print("\n")

c_zero = []
for i in range(len(to_concut[0])):
    c_zero.append(0)
c_new = np.concatenate((c, c_zero))
print(c_new)

#N,B
for i in range(len(A_new[0]) - len(A_new), len(A_new[0])):
    B = np.append(B, int(i))
for i in range(0, len(A_new[0]) - len(A_new)):
    N = np.append(N, int(i))

print(B+1)
print(N+1)

A_b = [[0] * len(B) for i in range(len(A_new))]

for i in range(len(B)):
    pointer_b = int(B[i])
    for j in range(len(A_new)):
        A_b[j][i] = A_new[j][pointer_b]

print(A_b)

A_n = [[0] * len(N) for i in range(len(A_new))]

for i in range(len(N)):
    pointer_n = int(N[i])
    for j in range(len(A_new)):
        A_n[j][i] = A_new[j][pointer_n]
        
print(A_n)

A_b_inverted = np.linalg.inv(A_b)
A_b_inverted

X_b = np.dot(A_b_inverted, b)
print(X_b)

C_b_transpose = []

for i in range(len(B)):
    pointer_c_b = int(B[i])
    C_b_transpose.append(c_new[pointer_c_b])
    
print(C_b_transpose) 

C_n_transpose = []

for i in range(len(N)):
    pointer_c_n = int(N[i])
    C_n_transpose.append(c_new[pointer_c_n])
    
print(C_n_transpose)

W_t = np.dot(C_b_transpose, A_b_inverted)
print(W_t)

S_N_transpose = np.subtract(C_n_transpose, np.dot(W_t, A_n))
print(S_N_transpose)

for i in range(len(S_N_transpose)):
    if S_N_transpose[i] < 0:
        P = np.append(P, i)
    else:
        Q = np.append(Q, i)

print(P+1)
print(Q+1)

for i in range(0, len(P)):
    L = np.append(L, 1)
print(L)

SnP = np.array([])
for i,x in enumerate(S_N_transpose):               
    if x < 0 :                        
        SnP = np.append(SnP,x)
print(SnP)

S0 = np.dot(L, SnP)
print(S0)

h1 = A_b_inverted
A_new[2][0]

list1 = P.tolist()              
list1 = list(map(int, list1)) 

A_for_P= A_new[:, list1]
A_for_P


h = 0
for i in range(0, len(P)):
    h += np.dot(A_b_inverted,A_for_P[:,i])
dB = np.dot(-L[i],h)

print(dB)

while not len(P) == 0:
    
    if all(i >= 0 for i in dB):
        if S0 == 0:
            print("Optimal Solution Found")
            break
    else:
        #calcutate leaving variable

        # Find out the minimum value and his indicator
        leaving = np.array([])
        min_leaving = [0]      
        for i in range(len(dB)):
            if dB[i] >= 0:
                leaving = np.append(leaving,999999999999)
            elif dB[i] < 0:
               temp = X_b[i] / dB[i]
               leaving = np.append(leaving,-temp)
        print("α list: ", leaving, "\n")

        min_leaving = np.amin(leaving)                        
        r = np.where(leaving == np.amin(leaving))             
        r = int(r[0][0])
        if math.isinf(r):
            print("Linear Problem is unbounded")
            break
        print("Index of min α είναι ο r: ", r+1, "\n")

        for i,x in enumerate(B):                     
            if i == r:                              
                K = int(x)   
        print("k =", K+1)
        
        #calculate incoming variable
        #Takes the row index that r has stored from A_b^-1
        Br =np.array([])
        for i,x in enumerate(to_concut):
            if i == r:                         
                Br = np.append(Br,x)
        print("Br: ", Br, "\n")

        # Calculate to Hrp. 
        for i in range(0, len(P)):
            HrP = np.dot(Br,A_for_P) 
        print("Hr_p: ", HrP, "\n")

        list2 = Q.tolist()                      
        list2= list(map(int, list2))           
        A_for_Q = A_new[:, list2]
        print(A_for_Q, "\n")

        for i in range(0, len(Q)):
            HrQ = np.dot(Br,A_for_Q)
        print("Hr_q: ", HrQ, "\n")    
        

        #THETA 1
        Sp = SnP

        theta1 = np.array([])
        min_theta1 = [0]
        for i in range(len(P)):
            if HrP[i] > 0:
                temp3 = -(Sp[i]) / HrP[i]
                theta1 = np.append(theta1, temp3)
            else:                                                
                theta1 = np.append(theta1, 999999999999)
        print("θ1 list: ", theta1, "\n")

        min_theta1 = np.amin(theta1)                 
        t1 = np.where(theta1 == np.amin(theta1))     
        t1 = int(t1[0])
        print("min θ1: ", min_theta1, "\n")
        print("Index of min θ1: ", t1+1)

        #THETA 2
        SnQ = np.array([])
        for i,x in enumerate(S_N_transpose):               
            if x >= 0: 
                SnQ = np.append(SnQ,x)
        print("SnQ: ", SnQ, "\n")

        theta2 = np.array([])
        min_theta2 = [0]
        for i in range(len(Q)):
            if HrP[i] > 0:
                temp4 = -(SnQ[i]) / HrQ[i]
                theta2 = np.append(theta2, temp4)
            else:                                                
                theta2 = np.append(theta2, 999999999999)
        print("θ2 list: ", theta2, "\n")

        min_theta2 = np.amin(theta2)                 
        t2 = np.where(theta2 == np.amin(theta2))     
        t2 = int(t2[0])
        print("min θ2: ", min_theta2, "\n")
        print("Index of min θ2: ", t2+1)
        
        l1=0
        for i in range(len(Sp)):   
            if t1 == i:
                  l1 = t1
        print("l1 = ", l1+1, "\n")

        p = np.array([]) 
        for i,x in enumerate(P):                 
            if l1 == i :                        
                p = np.append(p,x)
        print("p = ", p+1)

        l2=0
        for i in range(len(SnQ)): 
            if t2 == i:
                  l2 = t2
        print("l2 = ", l2+1, "\n")

        q = np.array([]) 
        for i,x in enumerate(Q):                 
            if l2 == i :                        
                q = np.append(q,x)
        print("q = ", q+1)
 
        if min_theta1 <= min_theta1:
          l = p                         
        else:                                   
          l = q

        print("l = ", l+1, "\nΆρα η μεταβλητή X", l+1, "εισέρχεται στη βάση")
       

        for i,x in enumerate(B):           
            if i == r :                         
                B[i] = l  
        print("B =", B+1)

        if min_theta1 <= min_theta2:
            for i,x in enumerate(P):
                if i == t1:                              
                    Q = np.append(Q, K)                  
                    P = np.delete(P,t1)   
        elif min_theta1 > min_theta2:
            for i,x in enumerate(Q):
                if i==t2:
                    Q[i] = K
        print("P: ", P+1, "\n")
        print("Q: ", Q+1, "\n")

        N = np.array([])
        for i in range(len(P)):
            N = np.append(N, P[i])
        for i in range(len(Q)):
            N = np.append(N, Q[i])
        print("N: ", N+1, "\n")

        A_l = np.array([])

        for j in range(len(A_new[0])):
            if j == l:
                for i in range(len(A_new)):
                    A_l = np.append(A_l, A_new[i][j])

        print("A_l: ", A_l, "\n")

        E = np.identity(len(A_b))

        for j in range(len(E[0])):
            if j == r:
                for i in range(len(E)):
                    E[i][j] = A_b_inverted[i][j] * A_l[i]
        print("E: ", E, "\n")

        A_b = [[0] * len(B) for i in range(len(A_new))]

        for i in range(len(B)):
            pointer_b = int(B[i])
            for j in range(len(A_new)):
                A_b[j][i] = A_new[j][pointer_b]

        print("A_b: ", A_b, "\n")

        A_b_inverted = np.linalg.inv(A_b)
        print("A_b_inverted: ", A_b_inverted, "\n")

        E_inv = np.identity(len(A_b))

        for j in range(len(A_b_inverted[0])):
            if j == r:
                for i in range(len(A_b_inverted)):
                    E_inv[i][j] = -(A_b_inverted[i][j])
        print("E inveted: ", E_inv, "\n")

        dB = np.dot(E_inv,dB)
        print("dB: ", dB, "\n")

        A_n = [[0] * len(N) for i in range(len(A_new))]

        for i in range(len(N)):
            pointer_n = int(N[i])
            for j in range(len(A_new)):
                A_n[j][i] = A_new[j][pointer_n]

        print("A_n: ", A_n, "\n")

        X_b = np.dot(A_b_inverted, b)
        print("X_b: ", X_b, "\n")

        C_b_transpose = []

        for i in range(len(B)):
            pointer_c_b = int(B[i])
            C_b_transpose.append(c_new[pointer_c_b])

        print("C_b_transpose: ", C_b_transpose, "\n")

        W_t = np.dot(C_b_transpose, A_b_inverted)
        print("W_t: ", W_t, "\n")

        C_n_transpose = []

        for i in range(len(N)):
            pointer_c_n = int(N[i])
            C_n_transpose.append(c_new[pointer_c_n])

        print("C_n_transpose: ", C_n_transpose, "\n")

        S_N_transpose = np.subtract(C_n_transpose, np.dot(W_t, A_n))
        print("S_N_transpose: ", S_N_transpose, "\n")

        L = np.array([])
        for i in range(0, len(P)):
            L = np.append(L, 1)
        print("L: ", L, "\n")

        SnP = np.array([])
        for i,x in enumerate(S_N_transpose):               
            if x < 0 :
                SnP = np.append(SnP,x)
        print("SnP: ", SnP, "\n")
        
        S0 = np.dot(L, SnP)
        print("S0: ", S0, "\n")

        
        if l in P:
            #for i in range(len(dB)):
                #if i == l:
            for i in range(len(P)):
                if P[i] == l:
                    ind = i
            dB[r] = np.add(dB[r],L[ind])

        print("dB: ", dB, "\n")

        print("=============================================")