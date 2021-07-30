import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_riemann
import scipy


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
    
def m_estimator(X,u,threshold=1e-6,max_iter=500):
    #for mle : u(x) = 1
    #for tyler: u(x) = n/x ## add power information
    #for huber of parameter r : u(x) = k if (|x|<r) ;else kr/|x|
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


def huber(x,k,r):
    if x < r:
        return k
    else:
        return k*r/x
    

    
def huber_param_first_method(X_trials,y):
    scms =  np.zeros((X_trials.shape[0],X_trials.shape[1],X_trials.shape[1]))
    for i in range(X_trials.shape[0]):
        scms[i,:,:]= scm(X_trials[i,:,:])
        
    mean = 0
    N = len(scms)
    for i in range(1,N):
        for j in range(i):
            mean += distance_riemann(scms[i,:,:],scms[j,:,:])
    mean = 2*mean/(N*(N-1))
    return mean


def huber_param_second_method(X_trials,y):
    y = np.asarray(y)
    scms =  np.zeros((X_trials.shape[0],X_trials.shape[1],X_trials.shape[1]))
    for i in range(X_trials.shape[0]):
        scms[i,:,:]= scm(X_trials[i,:,:])
        
    classes = np.unique(y)
    covcenters = [mean_riemann(scms[y==l]) for l in classes]
    r = 0
    for k in range(len(classes)):
        mindist = np.inf
        for i in range(len(scms)):
            dist = distance_riemann(scms[i,:,:],covcenters[k])
            if dist < mindist:
                mindist = dist
        if r < mindist:
            r= mindist
    return r     

def tyler_adaptive(X_trials):
    N,n,m = X_trials.shape #m is the number of points "(per session:(32,32,768) => m =768
    u = lambda x : n/x
    tyler_covs = []
    for k in range(N):
        X = X_trials[k,:,:]
        cov_1 =  m_estimator(X,u)
        X = X- mean(X)
        tau = []
        for i in range(m):
            Xi = X[:,i]
            tau.append(Xi.T@np.linalg.pinv(cov_1)@Xi)
        tau_mean =0
        for s in tau:
            tau_mean +=np.log(s)
        tau_mean = tau_mean/m
        tyler_covs.append(np.exp(tau_mean)*cov_1)
    tyler_covs = np.asarray(tyler_covs)
    return tyler_covs

def huber_adaptive(X_trials, clean_prop = 0.9):
    N,n,m = X_trials.shape
    new_X_trials = []
    scms = [scm(X_trials[k,:,:]) for k in range(N)]
    for k in range(N): # for the k^th trial
        X = X_trials[k,:,:]-mean(X_trials[k,:,:])
        all_arg = [X[:,i].T@np.linalg.pinv(scms[k])@X[:,i] for i in range(m)]
        indx = sorted(range(len(all_arg)), key=lambda k: all_arg[k])
        nb_clean_data = int(m*clean_prop)
        param = all_arg[indx[nb_clean_data]]
    u = lambda x : huber(x,1,param)
    huber_covs = [m_estimator(X_trials[k,:,:],u) for k in range(N)]
    huber_covs = np.asarray(huber_covs)
    return huber_covs
    
    
def huber_non_adaptive(X_trials,q):
    N,n,m = X_trials.shape
    param = 0.5*scipy.stats.chi2.ppf(q,2*n)
    u =  lambda x : huber(x,1,param)
    huber_covs = [m_estimator(X_trials[k,:,:],u) for k in range(N)]
    huber_covs = np.asarray(huber_covs)
    return huber_covs
    
                         
def covariances(X_trials,estimator,param = None, adaptive= True, clean_prop= 0.9, ddl = 5, labels = None):
    N,n,m = X_trials.shape
    
    if estimator=="scm":
        res = [scm(X_trials[k,:,:]) for k in range(N)]
        res = np.asarray(res)
    
    if estimator=="tyler":
        if adaptive:
            res = tyler_adaptive(X_trials)
        else :
            u = lambda x: n/x
            res = [m_estimator(X_trials[k,:,:],u) for k in range(N)]
            res = np.asarray(res)

    if estimator=="huber":
        if param == None:
            if adaptive:
                res = huber_adaptive(X_trials,clean_prop)
            else:
                res = huber_non_adaptive(X_trials,clean_prop)
                         
    if estimator=="student":
        u = lambda x : (n + ddl/2)/(ddl/2+x)
        res = [m_estimator(X_trials[k,:,:],u) for k in range(N)]
        res= np.asarray(res)
    
    return res
        

class Covariances(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator='scm',param=None, adaptive = True,clean_prop = 0.9, ddl = 5):
        self.estimator = estimator
        self.param = param
        self.adaptive = adaptive
        self.clean_prop = clean_prop
        self.ddl = ddl

    def fit(self, X, y=None):
        self.labels = y
        return self

    def transform(self, X):
        covmats = covariances(X, self.estimator,self.param,self.adaptive,self.clean_prop,self.ddl,self.labels)
        return covmats

