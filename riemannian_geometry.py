#!/usr/bin/env python
# coding: utf-8

# $$F(P) : = \frac{1}{2}d(P,Q)^2 = \frac{1}{2}\|\log(P^{-1}Q)\|^2_{fro}$$
# $$\nabla F(P) = P \log(P^{-1}Q)$$

# In[ ]:


import numpy as np
import scipy
from scipy.linalg import expm
from sklearn.datasets import make_spd_matrix


def verify_SP(X):
    """returns true is X is symmetric and positive matrix ie X.T=X and for u !=0, u.T@X@u > 0"""
    if np.all(X.T == X):
        eigenvalues,_ = np.linalg.eig(X)
        for x in eigenvalues:
            if x <=0:
                return False
        return True
    else:
        return False
    
def verify_SDP(X):
    """returns true is X is symmetric and positive matrix ie X.T=X and for u , u.T@X@u => 0"""
    if np.all(X.T == X):
        eigenvalues,_ = np.linalg.eig(X)
        for x in eigenvalues:
            if x <0:
                return False
        return True
    else:
        return False

def distance_riemann(X,Y):
    #assert verify_SP(X)==True
    #assert verify_SP(Y)==True
    assert X.shape == Y.shape
    eigenvalues,_ = np.linalg.eig(np.linalg.pinv(X)@Y)
    log_eigenvalues = np.log(eigenvalues)
    dist = np.sqrt(np.sum(log_eigenvalues**2))
    return dist

def logm(X):
    if verify_SP(X):
        v,w = np.linalf.eig(X)
        diagonal_log = np.diag(np.log(v))
        return w@diagonal_log @np.linalg.pinv(w) 
    else:
        return scipy.linalg.logm(X)
        
def sqrtm(X):
    """returns X**(1/2)"""
    assert verify_SDP(X)
    v,w = np.linalf.eig(X)
    diagonal_sqrt = np.diag(np.sqrt(v))
    return w@diagonal_sqrt @np.linalg.pinv(w) 
    
def invsqrtm(X):
    """returns X**(-1/2)"""
    assert verify_SP(X)
    v,w = np.linalf.eig(X)
    diagonal_invsqrt = np.diag(1/np.sqrt(v))
    return np.linalg.pinv(w) @diagonal_invsqrt @w
   
def exp_riemann(X,Y):
    """ exp_X(Y) = X exp(X^{-1}Y) = X^{1/2} exp(X^{-1/2} Y X^{-1/2}) X^{1/2}"""
    return X@expm(np.linalg.pinv(X)@Y)
    
def log_riemann(X,Y):
    """ log_X(Y) = log(X^{-1}Y) = X^{1/2} log(X^{-1/2} Y X^{-1/2}) X^{1/2}"""
    return X@logm(np.linalg.pinv(X)@Y)

def inner_riemann(X,A,B):
    invX = np.linalg.pinv(X)
    return np.matrix.trace(invX@A@invX@B)

def total_distance(X,Ys):
    M = len(Ys)
    s = 0
    for k in range(M):
        s+=distance_riemann(X,Ys[k])**2
    return 0.5*s

##Question: convex/strong-convex over geodesics?

def mean_riemann(Ys, beta = 0.5, alpha =0.5,gamma=0.5, threshold=1e-3,max_iter=100):
    """Ys: list of SP matrices of same shape N*N
    returns the argmin of sum_{i} d(X,Ys_i)^2 using GD with backtrack line search"""
    M = len(Ys)
    if type(Ys) !=list:
        newYs= [Ys[k,:,:] for k in range(M)]
        Ys = newYs
        
    N = Ys[0].shape[0]
    invYs = [np.linalg.pinv(Ys[k]) for k in range(M)]
    Z = np.random.rand(N,N).astype(Ys[0].dtype)
    X0 = np.eye(N).astype(Ys[0].dtype) + (Z@Z.T)
    i =0
    error =np.inf
    X = X0
    while (i<max_iter) and (error>threshold):
        grad = logm(invYs[0]@X)
        for k in range(1,M):
            grad += logm(invYs[k]@X)
        grad = X@grad
        
        #backtracking line search
        delta_x = 0.5*(grad+grad.T)
        eigval_max_X = np.max(np.linalg.eig(X)[0])
        eigval_min_delta= np.min(np.linalg.eig(delta_x)[0])
        if eigval_min_delta <0 : #any value ensure the SP
            h = 1
        else:
            h = min(1,eigval_max_X /eigval_min_delta) #heursitic, to not start from large values
           
        j = 0
        while not(verify_SP(X-h*delta_x)):
            j += 1
            h = (gamma**(i//10+1))*h 
            #we searched the highest h0 fulfilling the SP criterion; any value lower than h0 satisfies then the Sp criterion
                  

        while (total_distance(X-h*delta_x,Ys) < total_distance(X,Ys) - alpha*h*np.linalg.norm(grad,'fro')**2):
            j += 1
            h = beta*h
        
        print(i,j)
        X = X - h* delta_x   
        i += 1
        error = np.linalg.norm(grad,'fro')
        print(error)
    return X


    

