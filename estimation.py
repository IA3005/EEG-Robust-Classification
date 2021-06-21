#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def scatter_matrix(X):
    """
    X : square matrix of shape N*M
    returns a matrix of shape N*N
    """
    _,m = X.shape
    U =np.ones((m,1))
    return X@(np.eye(m)-(1/m)*U@U.T)@X.T
    
def scm(X):
    #unbiased
    _,m = X.shape
    return scatter_matrix(X)/(m-1)

def mle(X):
    #baised
    n,m = X.shape
    return scatter_matrix(X)/m

def mean(X):
    """returns the mean of shape (N,1)"""
    n,m = X.shape
    U =np.ones((m,1)) 
    emp_mean = (X@U)/m
    emp_mean = emp_mean.reshape((n,1))
    return emp_mean

def huber(x,r):
    d = np.abs(x)
    if d <= r:
        return 0.5*d**2
    else:
        return r*(d-0.5*r)
    
def pseudo_huber(x,r):
    d = np.abs(x)
    return (r**2)*(np.sqrt(1+(d/r)**2)-1)
    
    
def m_estimator(X,u,threshold=1e-6,max_iter=500):
    #for mle : u(x) = 1
    #for tyler: u(x) = n/x
    #for huber of parameter r : u(x) = k/r if (|x|<r) ;else k/|x|
    n,m = X.shape
    X_prime = X - mean(X) 
    cov_0 = scm(X) #possible init with the scm
    cov = cov_0 #of shape (n,n)
    error = np.inf
    i=0
    while (error > threshold) and (i < max_iter):
        i +=1
        inv_cov = np.linalg.pinv(cov)
        U = np.diag([u(X_prime[:,i].T@inv_cov@X_prime[:,i]) for i in range(m)])
        new_cov = (X_prime@(U@X_prime.T))/m
        error = np.linalg.norm(cov-new_cov)
        cov = new_cov
    return cov


def tyler(x,params):
    r,k = params
    if x < r:
        return k/r
    else:
        return k/x
    
    
def covariance(X,estimator,tyler_params = None):
    if estimator=="scm":
        return scm(X)
    if estimator=="huber":
        n,_ = X.shape
        u = lambda x : n/x
    if estimator=="tyler":
        assert tyler_params != None
        u = lambda x : tyler(x,tyler_param)
    return m_estimator(X,u)
        
def covariances(X_trials,estimator,tyler_params = None):
    assert len(X_trials.shape)==3 #(n_trials,N,M)
    res = np.zeros((X_trials.shape[0],X_trials.shape[1],X_trials.shape[1]))
    for i in range(X_trials.shape[0]):
        res [i,:,:]= covariance(X_trials[i,:,:],estimator,tyler_params = None)
    return res
        


