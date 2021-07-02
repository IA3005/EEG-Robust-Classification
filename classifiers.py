#from riemannian_geometry import distance_riemann, mean_riemann

import numpy as np
from scipy import stats

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.extmath import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_riemann
from riemannian_geometry import project,reverse_project,verify_SDP,mean_euclidian


class MDM(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Classification by Minimum Distance to Mean.
    """
    
    def __init__(self, n_jobs=1, robustify=False):
        """Init."""
        # store params for cloning purpose
        self.n_jobs = n_jobs
        self.robustify = robustify
        
        

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
            self.covmeans_ = [mean_riemann(X[y == ll]) for ll in self.classes_]
        else:
            self.covmeans_ = Parallel(n_jobs=self.n_jobs)(
                delayed(mean_riemann)(X[y == ll]) for ll in self.classes_)

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

    """Tangent space project TransformerMixin.
    """

    def __init__(self,reference=None,robustify=False):
        self.reference= reference
        self.robustify = robustify

    def fit(self, X, y=None):
        """Fit (estimates) the reference point.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray | None (default None)
            Not used, here for compatibility with sklearn API.
        sample_weight : ndarray | None (default None)
            weight of each sample.
        Returns
        -------
        self : TangentSpace instance
            The TangentSpace instance.
        """
        # compute mean covariance
        if self.reference == None:
            self.reference = mean_euclidian(X,self.robustify)
        else: 
            assert verify_SDP(self.reference),"The given reference point is not SDP"
        return self

    
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
        ts  = project(self.reference,X)
        return ts


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
        return reverse_project(self.reference,X)

    

class TSclassifier(BaseEstimator, ClassifierMixin):

    """Classification in the tangent space.
    """

    def __init__(self,clf=LogisticRegression(),reference=None,robustify=False):
        """Init."""
        self.robustify = robustify
        self.reference = reference
        self.clf = clf

        if not isinstance(clf, ClassifierMixin):
            raise TypeError('clf must be a ClassifierMixin')

    def fit(self, X, y):
        """Fit TSclassifier.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.
        Returns
        -------
        self : TSclassifier. instance
            The TSclassifier. instance.
        """
        self.classes_ = np.unique(y)
        ts = TangentSpace(reference=self.reference,robustify=self.robustify)
        self._pipe = make_pipeline(ts, self.clf)
        self._pipe.fit(X, y)
        return self

    def predict(self, X):
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
        return self._pipe.predict(X)

    def predict_proba(self, X):
        """get the probability.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        pred : ndarray of ifloat, shape (n_trials, n_classes)
            the prediction for each trials according to the closest centroid.
        """
        return self._pipe.predict_proba(X)


