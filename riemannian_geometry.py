# $$F(P) : = \frac{1}{2}d(P,Q)^2 = \frac{1}{2}\|\log(P^{-1}Q)\|^2_{fro}$$
# $$\nabla F(P) = P \log(P^{-1}Q)$$


import numpy as np
import scipy
from sklearn.datasets import make_spd_matrix
from scipy.linalg import eigvalsh, eigh
from pyriemann.utils.base import logm, sqrtm,invsqrtm,expm
from pyriemann.utils.distance import distance_riemann


def verify_SDP(X):
    """returns true is X is symmetric and positive matrix ie X.T=X and for u !=0, u.T@X@u > 0"""
    if np.all(X.T == X):
        eigenvalues = eigvalsh(X)
        for x in eigenvalues:
            if x <=0:
                return False
        return True
    else:
        return False
    
def verify_SSDP(X):
    """returns true is X is symmetric and positive matrix ie X.T=X and for u , u.T@X@u => 0"""
    if np.all(X.T == X):
        eigenvalues = eigvalsh(X)
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
        for i in range(j+1):
            if i==j:
                v.append(A[i,j])
            else:
                v.append(np.sqrt(2)*A[i,j])
    v = np.asarray(v)
    return v
    
def unvectorize(v):
    m = len(v)
    #n(n+1)/2 = m then n^2+n-2m=0
    n = int((np.sqrt(1+8*m)-1)/2)
    A = np.zeros((n,n))
    k= 0
    for j in range(n):
        for i in range(j+1):
            if i==j:
                A[i,j] = v[k]
            else: 
                A[i,j] = v[k]/np.sqrt(2)
                A[j,i] = A[i,j]
            k += k
    return A

"""
def distance_riemann(X,Y):
    eigenvalues = eigvalsh(X, Y)
    log_eigenvalues = np.log(eigenvalues)
    dist = np.sqrt(np.sum(log_eigenvalues**2))
    return dist

def logm(X):
    v,w = eigh(X,check_finite=False)
    diagonal_log = np.diag(np.log(v))
    return w@diagonal_log @np.linalg.pinv(w) 
    

def expm(X):
    v,w = eigh(X,check_finite=False)
    diagonal_exp = np.diag(np.exp(v))
    return w@diagonal_exp @np.linalg.pinv(w) 
     
    
def sqrtm(X):
    #returns X**(1/2)
    #assert verify_SSDP(X)
    v,w = eigh(X,check_finite=False)
    diagonal_sqrt = np.diag(np.sqrt(v))
    return w@diagonal_sqrt @np.linalg.pinv(w) 
    
def invsqrtm(X):
    #returns X**(-1/2)
    #assert verify_SDP(X)
    v,w = eigh(X,check_finite=False)
    diagonal_invsqrt = np.diag(1/np.sqrt(v))
    return np.linalg.pinv(w) @diagonal_invsqrt @w
"""   
def exp_riemann(X,Y):
    """ exp_X(Y) = X exp(X^{-1}Y) = X^{1/2} exp(X^{-1/2} Y X^{-1/2}) X^{1/2}"""
    Xsqrt = sqrtm(X)
    Xinvsqrt = invsqrtm(X)
    return Xsqrt@expm(Xinvsqrt@Y@Xinvsqrt)@Xsqrt
    
def log_riemann(X,Y):
    """ log_X(Y) = X log(X^{-1}Y) = X^{1/2} log(X^{-1/2} Y X^{-1/2}) X^{1/2}"""
    Xsqrt = sqrtm(X)
    Xinvsqrt = invsqrtm(X)
    return Xsqrt@logm(Xinvsqrt@Y@Xinvsqrt)@Xsqrt

def inner_riemann(X,A,B):
    invX = np.linalg.pinv(X)
    return np.matrix.trace(invX@A@invX@B)


def mean_riemann(covmats, tol=10e-9, maxiter=50, init=None, u_prime=lambda x : 1):
    
    Nt, Ne, Ne = covmats.shape
    if init is None:
        C = np.mean(covmats, axis=0)
    else:
        C = init
    k = 0
    nu = 1.0
    tau = np.finfo(np.float64).max
    crit = np.finfo(np.float64).max
    # stop when J<10^-9 or max iteration = 50
    while (crit > tol) and (k < maxiter) and (nu > tol):
        k = k + 1
        C12 = sqrtm(C)
        Cm12 = invsqrtm(C)
        J = np.zeros((Ne, Ne))

        for i in range(Nt):
            tmp = (Cm12 @ covmats[i, :, :])@ Cm12
            if type(u_prime(1))==list:
                J += logm(tmp)* u_prime(distance_riemann(C,covmats[i,:,:])**2)[i]/Nt
            else:
                J += logm(tmp)* u_prime(distance_riemann(C,covmats[i,:,:])**2)/Nt

        crit = np.linalg.norm(J, ord='fro')
        h = nu * crit
        C = np.dot(np.dot(C12, expm(nu * J)), C12)
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu

    return C



def project(reference,Ys):
    if type(Ys) !=list:
        newYs = [Ys[i,:,:] for i in range(Ys.shape[0])]
        Ys= newYs
    res = []
    for i in range(len(Ys)):
        proj_cov = log_riemann(reference,Ys[i]) #Ys[i] of shape(n_trials,n)
        vect_proj_cov = vectorize(proj_cov)
        res.append(res)
    ts = np.zeros((len(res),len(res[0]))) #ts of shape (n_trials, n*(n+1)/2)
    for i in range(len(res)):
        ts[i,:] =res[i]
    return ts

def reverse_project(reference,ts):
    if type(ts) !=list:
        newts = [ts[i,:] for i in range(ts.shape[0])]
        ts= newts
    res = []
    for i in range(len(Ys)):
        unvec_t = unvectorize(ts[i]) #from length m  to shape (n,n) s.t. n(n+1)/2=m
        unproj_cov = exp_riemann(reference,unvec_t) #shape(n,n)
        res.append(unproj_cov)
                  
    X = np.zeros((len(res),res[0].shape[0],res[0].shape[0])) #ts of shape (n_trials, n*(n+1)/2)
    for i in range(len(res)):
        X[i,:,:] =res[i]
    return X
    

    

             
         
                          