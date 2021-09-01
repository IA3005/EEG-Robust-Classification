from riemannian_geometry import mean_riemann
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.extmath import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed
from pyriemann.utils.distance import distance_riemann
#from pyriemann.utils.mean import mean_riemann
from riemannian_geometry import project,reverse_project,verify_SDP
from pyriemann.utils.tangentspace import tangent_space, untangent_space



class MDM(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Classification by Minimum Distance to Mean.
    """
    
    def __init__(self, n_jobs=1, u_prime=lambda x :1):
        """Init."""
        # store params for cloning purpose
        self.n_jobs = n_jobs
        self.u_prime = u_prime
        
        

    def fit(self, X, y):
        """Fit (estimates) the centroids.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.
        sample_weight : None | ndarray shape (n_trials, 1)
            the weights of each sample. if None, each sample is treated with
            equal weights.
        Returns
        -------
        self : MDM instance
            The MDM instance.
        """
        self.classes_ = np.unique(y)

        y = np.asarray(y)
        if self.n_jobs == 1:
            self.covmeans_ = [mean_riemann(X[y == ll],u_prime=self.u_prime) for ll in self.classes_]
        else:
            self.covmeans_ = Parallel(n_jobs=self.n_jobs)(
                delayed(mean_riemann)(X[y == ll],u_prime=self.u_prime) for ll in self.classes_)

        return self

    def _predict_distances(self, covtest):
        """Helper to predict the distance. equivalent to transform."""
        Nc = len(self.covmeans_)
        dist = np.zeros((covtest.shape[0],Nc)) #shape= (n_trials,n_classes)
        for j in range(covtest.shape[0]):
            if self.n_jobs == 1:
                dist_j = [distance_riemann(covtest[j,:,:], self.covmeans_[m]) for m in range(Nc)]
            else:
                dist_j = Parallel(n_jobs=self.n_jobs)(delayed(distance_riemann)(covtest[j,:,:], self.covmeans_[m])for m in range(Nc))
            dist_j = np.asarray(dist_j)
            dist[j,:] = dist_j

        return dist

    def predict(self, covtest):
        """get the predictions.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        dist = self._predict_distances(covtest)
        preds = []
        n_trials,n_classes = dist.shape
        for i in range(n_trials):
            preds.append(self.classes_[dist[i,:].argmin()])
        preds = np.asarray(preds)
        return preds

    def transform(self, X):
        """get the distance to each centroid.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        dist : ndarray, shape (n_trials, n_classes)
            the distance to each centroid according to the metric.
        """
        return self._predict_distances(X)

    def fit_predict(self, X, y):
        """Fit and predict in one function."""
        self.fit(X, y)
        return self.predict(X)

    def predict_proba(self, X):
        """Predict proba using softmax.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        prob : ndarray, shape (n_trials, n_classes)
            the softmax probabilities for each class.
        """
        return softmax(-self._predict_distances(X)**2)


    
class TangentSpace(BaseEstimator, TransformerMixin):
    
    def __init__(self, tsupdate=False, u_prime = lambda x:1):
        self.tsupdate = tsupdate
        self.u_prime = u_prime

    def fit(self, X, y=None):
        # compute mean covariance
        self.reference_ = mean_riemann(X ,u_prime=self.u_prime)
        return self

    def _check_data_dim(self, X):
        """Check data shape and return the size of cov mat."""
        shape_X = X.shape
        if len(X.shape) == 2:
            Ne = (np.sqrt(1 + 8 * shape_X[1]) - 1) / 2
            if Ne != int(Ne):
                raise ValueError("Shape of Tangent space vector does not"
                                 " correspond to a square matrix.")
            return int(Ne)
        elif len(X.shape) == 3:
            if shape_X[1] != shape_X[2]:
                raise ValueError("Matrices must be square")
            return int(shape_X[1])
        else:
            raise ValueError("Shape must be of len 2 or 3.")

    def _check_reference_points(self, X):
        """Check reference point status, and force it to identity if not."""
        if not hasattr(self, 'reference_'):
            self.reference_ = np.eye(self._check_data_dim(X))
        else:
            shape_cr = self.reference_.shape[0]
            shape_X = self._check_data_dim(X)

            if shape_cr != shape_X:
                raise ValueError('Data must be same size of reference point.')

    def transform(self, X):
        """Tangent space projection.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        ts : ndarray, shape (n_trials, n_ts)
            the tangent space projection of the matrices.
        """
        self._check_reference_points(X)
        if self.tsupdate:
            Cr = mean_riemann(X ,u_prime=self.u_prime)
        else:
            Cr = self.reference_
        return tangent_space(X, Cr)

    def fit_transform(self, X, y=None):
        """Fit and transform in a single function.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray | None (default None)
            Not used, here for compatibility with sklearn API.
        Returns
        -------
        ts : ndarray, shape (n_trials, n_ts)
            the tangent space projection of the matrices.
        """
        # compute mean covariance
        self._check_reference_points(X)
        self.reference_ = mean_riemann(X ,u_prime=self.u_prime)
        return tangent_space(X, self.reference_)

    def inverse_transform(self, X, y=None):
        """Inverse transform.
        Project back a set of tangent space vector in the manifold.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_ts)
            ndarray of SPD matrices.
        y : ndarray | None (default None)
            Not used, here for compatibility with sklearn API.
        Returns
        -------
        cov : ndarray, shape (n_trials, n_channels, n_channels)
            the covariance matrices corresponding to each of tangent vector.
        """
        self._check_reference_points(X)
        return untangent_space(X, self.reference_)