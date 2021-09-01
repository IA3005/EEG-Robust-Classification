import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_riemann
import scipy

   
def scm(X):
    #unbiased
    n,m = X.shape
    #X = X - mean(X)
    return np.dot(X,X.T)/(m-1)

def mle(X):
    #baised
    n,m = X.shape
    #X = X - mean(X,)
    return np.dot(X,X.T)/m

def mean(X):
    """returns the mean of shape (n,m)"""
    Y = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Y[i,:] = X[i,:] - np.mean(X[i,:])
    #Y = np.expand_dims(np.mean(X,axis=1),axis=1)@np.ones((1,X.shape[1])
    return Y
    
def m_estimator(X,u,threshold=1e-10,max_iter=500,normalize=True):
    #for mle : u(x) = 1
    #for tyler: u(x) = n/x ## add power information
    #for huber of parameter r : u(x) = k if (|x|<r) ;else kr/|x|
    n,m = X.shape
    #X = X - mean(X) 
    #Z = np.random.rand(n,m)
    #cov_0 = Z@Z.T+np.random.rand()*np.eye(n)
    cov_0 = scm(X) #possible init with the scm
    cov = cov_0 #of shape (n,n)
    error = np.inf
    errors = []
    i=0
    while (error > threshold) and (i < max_iter):
        i +=1
        inv_cov = np.linalg.pinv(cov)
        U = np.diag([u(X[:,j].T@inv_cov@X[:,j]) for j in range(m)])
        new_cov = (X@(U@X.T))/m
        error = np.linalg.norm(cov-new_cov)
        errors.append(error)
        cov = new_cov 
    if normalize:
        cov = cov/np.trace(cov)
    return cov,errors


def huber(x,k,r):
    if x < r:
        return k
    else:
        return k*r/x
        

def tyler_adaptive(X_trials,normalize=False):
    N,n,m = X_trials.shape #m is the number of points "(per session:(32,24,768) => m =768
    u = lambda x : n/x
    tyler_covs = []
    all_errors = []
    for k in range(N):
        X = X_trials[k,:,:]
        cov , errors =  m_estimator(X,u,normalize=False)
        all_errors.append(errors)
        #X = X- mean(X)
        tau = []
        for i in range(m):
            Xi = X[:,i]
            tau.append(Xi.T@np.linalg.pinv(cov)@Xi)
        tau_mean =0
        for s in tau:
            tau_mean +=np.log(s)
        tau_mean = tau_mean/m
        new_cov = np.exp(tau_mean)*cov
        if normalize:
            new_cov = new_cov/np.trace(new_cov)
        tyler_covs.append(new_cov)
    return tyler_covs, all_errors

def huber_adaptive_param(X_trials, clean_prop = 0.9):
    N,n,m = X_trials.shape
    scms = [scm(X_trials[k,:,:]) for k in range(N)]
    params = []
    
    for k in range(N): # for the k^th trial
        #X = X_trials[k,:,:]-mean(X_trials[k,:,:])
        X = X_trials[k,:,:]
        all_arg = [X[:,i].T@np.linalg.pinv(scms[k])@X[:,i] for i in range(m)]
        indx = sorted(range(len(all_arg)), key=lambda k: all_arg[k])
        nb_clean_data = int(m*clean_prop)
        params.append(all_arg[indx[nb_clean_data]])

    return params
    
    
def huber_non_adaptive_param(X_trials,q):
    N,n,m = X_trials.shape
    param = 0.5*scipy.stats.chi2.ppf(q,2*n)
    return param
    
                         
def covariances(X_trials,estimator, clean_prop= 0.9, ddl = 5,check_conv = False):
    N,n,m = X_trials.shape
    
    if estimator=="scm":
        res = [scm(X_trials[k,:,:]) for k in range(N)]
        errors = []
    
    if estimator=="tyler adaptive":
        res,errors = tyler_adaptive(X_trials)
    
    if estimator=="tyler normalized non adaptive":
        u = lambda x: n/x
        res_errors = [m_estimator(X_trials[k,:,:],u) for k in range(N)]
        res = [res_errors[k][0] for k in range(N)]
        errors = [res_errors[k][1] for k in range(N)]
        for k in range(N):
            det = np.linalg.det(res[k])
            res[k] = res[k]/np.exp(np.log(det)/n)
    
    if estimator=="tyler non normalized non adaptive":
        u = lambda x: n/x
        res_errors = [m_estimator(X_trials[k,:,:],u) for k in range(N)]
        res = [res_errors[k][0] for k in range(N)]
        errors = [res_errors[k][1] for k in range(N)]
    
    if estimator=="huber adaptive":
        params = huber_adaptive_param(X_trials,clean_prop)
        res_errors = [m_estimator(X_trials[k,:,:],lambda x : huber(x,1,params[k])) for k in range(N)]
        res = [res_errors[k][0] for k in range(N)]
        errors = [res_errors[k][1] for k in range(N)]
         
    if estimator=="huber non adaptive":
        param = huber_non_adaptive_param(X_trials,clean_prop)
        res_errors = [m_estimator(X_trials[k,:,:],lambda x : huber(x,1,param)) for k in range(N)]
        res = [res_errors[k][0] for k in range(N)]
        errors = [res_errors[k][1] for k in range(N)]
        
    if estimator=="student":
        u = lambda x : (n + ddl/2)/(ddl/2+x)
        res_errors = [m_estimator(X_trials[k,:,:],u) for k in range(N)]
        res = [res_errors[k][0] for k in range(N)]
        errors = [res_errors[k][1] for k in range(N)]
    
    res= np.asarray(res)
    
    if check_conv:
        return res,errors
    else:
        return res
        

class Covariances(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator='scm',clean_prop = 0.9, ddl = 5,check_conv = False):
        self.estimator = estimator
        self.clean_prop = clean_prop
        self.ddl = ddl
        self.check_conv = check_conv
        self.errors = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.check_conv:
            covmats,errors = covariances(X, self.estimator,self.clean_prop,self.ddl, check_conv=True)
            self.errors = errors
        else:
            covmats = covariances(X, self.estimator,self.clean_prop,self.ddl, check_conv=False)
            
        return covmats

