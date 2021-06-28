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


def vectorize(A):
    """
    Input:
        A : a symmetric matrix of shape(n,n)
    Output:
        v : vectorized for of A ; v=[A11, sqrt(2)A12, A22, sqrt(2)A13, sqrt(2)A23, A33,...,Ann] of length n(n+1)/2
    """
    assert A.shape[0]==A.shape[1]
    n = A.shape[0]
    v = []
    for j in range(n):
        for i in range(j):
            if i==j:
                v.append(A[i,j])
            else:
                v.append(np.sqrt(2)*A[i,j])
    v = np.asarray(v)
    return v
    

    
    
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

def u(x,r = 1):
    if x < r:
        return x
    else:
        return r*np.exp(r-x)
    
def u_prime(x,r =1):
    if x < r:
        return 1
    else:
        return -r*exp(r-x)
    
    
def mean_riemann(Ys, h = 1e-3, threshold=1e-6,max_iter=500, robustify = False):
    """Ys: list of SP matrices of same shape N*N
    returns the argmin of sum_{i} d(X,Ys_i)^2 using GD with backtrack line search"""
    M = len(Ys)
    if type(Ys) !=list:
        newYs= [Ys[k,:,:] for k in range(M)]
        Ys = newYs
    iteration = 0
    error = np.inf
    
    x = Ys[0] #initial point = arithmetic mean
    for i in range(1,M):
        x += Ys[i]
    x = x/M
                                     
    while (error>threshold) and (iteration <max_iter):
        w = np.empty_like(Ys[0])
        for i in range(1,M):
            w += log_riemann(x,Ys[i])
            if robustify : 
                w = w*u_prime(distance_riemann(x,Ys[i]))
        w = w/M
        x_new = exp_riemann(x,h*w)
        error = np.linalg.norm(x-x_new,'fro')
        x = x_new
        iteration += 1

        
def arithmetic_mean(Ys):
    mean = Ys[0]
    for i in range(1,len(Ys)):
        mean += Ys[i]
    return mean/len(Ys)

def robust_arithmetic_mean(Ys):
    """TODO"""
    return Ys[0]

class TangentSpace():
    
    def __init__(self,reference=None,robustify=False):
        self.reference = reference
        self.robustify = robustify
    
    def fit(self,covs_list):
        if self.reference = None :
            if self.robustify ==False:
                self.reference = arithmetic_mean(covs_list)
            else:
                self.reference = robust_arithmetic_mean(covs_list)
        
        res = []
        for i in range(len(covs_list)):
            proj_cov = log_riemann(self.reference,covs_list[i])
            vect_proj_cov = vectorize(proj_cov)
            res.append(res)
        return res
    

             
         
                          