#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 08 17:07:02 2021

@author: alessio
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy import linalg as la

#------------------------------------------------------------------------------
#			ReLu and SiMEC functions
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#ReLu
def relu(x):
    result = -0.01*x
    if x > 0:
        result = x
    return result
#------------------------------------------------------------------------------
#ReLu derivative
def relu_derivative(x):
    result = -0.01
    if x > 0:
        result = 1
    return result
#------------------------------------------------------------------------------
#Composition of relu and a linear application
def fun(x,A):
    temp = A.dot(x)
    result = relu(temp)
    return result
#------------------------------------------------------------------------------
#Jacobian of a composition of relu and a linear application
def fun_jac(x,A):
    temp = A.dot(x)
    result = A*relu_derivative(temp)
    return result
#------------------------------------------------------------------------------
#Simec algorithm. Args:
# x : strating point
# A : matrix of weights
# steps : number of iterations
# delta : integration step
# epsilon: all the eigenvalues less than epsilon are considered as null.
def simec(x,A,steps,delta,epsilon):
    starting_point = x
    eq_class = []
    eq_class.append(x)
    euclidean_metric = np.identity(x.shape[0])
    sign = 1;
    old_eigen = [-1,0]
    #Compute the pullback metric to find dim(Ker f^*g) of the starting point 
    jac_mat = fun_jac(starting_point,A)
    jac_mat_t = np.transpose(jac_mat)
    g = np.einsum( "ij,jk,kl", jac_mat_t, euclidean_metric, jac_mat )
    #Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = la.eig(g)
    #Sort eigenvalues
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    #Find the eigenvectors < epsilon
    old_ker_dim = 0
    for i in eigenvalues:
        if (i < epsilon): old_ker_dim+=1
    for j in range(0,steps):
        #Find the eigenvectors < epsilon
        max_id = 0
        for i in eigenvalues:
            if (i < epsilon): max_id+=1
        #If the all the eigenvalues are greater than epsilon, exit the main loop
        if (max_id == 0): break
        #If the dimension of the kernel change, exit the loop
        if (max_id != old_ker_dim): break
        #Select a null eigenvector randomly
        idx = np.random.randint(0,max_id)
        #Proceed in the direction of a null eigenvector
        if np.dot(old_eigen,eigenvectors[:,idx])>0:
            sign = +1
        else:
            sign = -1
        new_point = starting_point + delta*eigenvectors[:,idx]*sign
        eq_class.append(new_point)
        starting_point = new_point
        old_eigen = eigenvectors[:,idx]
        #Compute the pullback metric
        jac_mat = fun_jac(starting_point,A)
        jac_mat_t = np.transpose(jac_mat)
        g = np.einsum( "ij,jk,kl", jac_mat_t, euclidean_metric, jac_mat )
        #Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = la.eig(g)
        #Sort eigenvalues
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]
    return eq_class

#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
#               Run the algoithm and plot the result
#------------------------------------------------------------------------------
#Set the matrix of weights and the starting point  
mat = np.matrix([1,-1])
sp = np.asarray([1.85,1.3])
#Run SiMEC  
eq_list = simec(sp,mat,10000,.001,0.00000001)
#Plot the result
xl, yl  = zip(*eq_list)
plt.scatter(xl,yl)
plt.plot()
plt.clf()

a = np.asarray([-1.85,1.3])
eq_list = simec(a,mat,10000,.001,0.00000001)
    
#print(diff)

#print(eq_list)
xl, yl  = zip(*eq_list)
plt.scatter(xl,yl)
plt.plot()
#plt.clf()
#Plot points



