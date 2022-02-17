from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np

class KNNClassifier(BaseEstimator):
    
    """K nearest neighbors classifier. Label assigned to each prediction
    point is the mode of the k nearest neighbors. When k is even, break any 
    ties using 1NN. e.g. k = 4 and the closest 4 points to point i (in X of
    predict()) have labels 'a', 'b', 'b', 'a' with respective distances 
    [1.4, 2, 3, 3.1]. We will assign label 'a' to point i.
    
    Parameters
    ----------
    dist : str, default='euclidean'
        The distance to calculate between a given prediction point and every other
        point in X_. 
        
    k: int, default=3
        The number of nearest neighbors used to predict a point's label.
        
    Attributes
    ----------
    X_ : pd.DataFrame, shape (n_samples, n_features)
        The training samples passed during fit().
    y_ : pd.Series, shape (n_samples,)
        The labels passed during fit().
    classes_ : ndarray, shape (n_classes,)
        The classes seen at fit().
    is_fitted_: bool
        Boolean for whether fit() has been called on the current instance, used by 
        check_is_fitted().
    n_features_in_: int
        Number of features seen during fit().
    """
    
    DISTS = {
        'manhattan': manhattan_distances,  # L1 
        'euclidean': euclidean_distances,  # L2
    }
    
    METRICS = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'auc': roc_auc_score
    }
    
    
    def __init__(self, dist='euclidean', k=3):        
        self.dist = dist
        self.k = k
        
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        
        """Fit the estimator. 
        
        Parameters
        ----------
        X : {pd.DataFrame, ndarray}, shape (n_samples, n_features)
            Training samples.
        y : {pd.Series, ndarray}, shape (n_samples,)
            Class labels.
            
        Returns
        -------
        self : object
            Returns self to allow chaining function calls after fit().
        """
        
        # Boilerplate stuff
        
        # Check X and y have correct shape
        X, y = check_X_y(X, y)
    
        # Extract unique labels
        self.classes_ = unique_labels(y)
        
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]
        
        self.X_ = pd.DataFrame(X)
        self.y_ = pd.Series(y)
        
        return self
    
    
    def predict(self, X):
        
        """Predict class labels for X.
        
        Parameters
        ----------
        X : {pd.DataFrame, ndarray}, shape (n_samples, n_features)
            Samples to get predictions.
            
        Returns
        -------
        y : pd.Series, shape (n_samples,)
            Predictions.
        """
        
        check_is_fitted(estimator=self, attributes='is_fitted_')
        
        X = pd.DataFrame(X)
        
        # dists is a (n_samples_X, n_samples_X_) matrix where each row contains 
        # distances between a point in X and every sample seen during fit (X_)
        dists = self.DISTS[self.dist](X, self.X_)
        ranks = np.argsort(dists, axis=1)
        
        y = []
        for i, rank in enumerate(ranks):
            rank = rank[:self.k]
            mode = self.y_.loc[rank].mode()
            if len(mode) > 1:
                # break voting ties by choosing 1NN
                nearest = self.y_.loc[np.argmin(dists[i])]
                y.append(nearest)
            else:
                y.append(mode.values[0])
                
        return pd.Series(y)
    
    
    def score(self, X, y, metric='accuracy', sample_weight=None, **kwargs):
        """Return metric score on the given test samples and labels.

        Parameters
        ----------
        X : {pd.DataFrame, ndarray}, shape (n_samples, n_features)
            Test samples.
        y : {pd.Series, pd.DataFrame, ndarray}, shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.
        metric: str, default='accuracy'
            Metric to calculate the score.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights for samples in X.
        Returns
        -------
        score : float
            Score of self.predict(X) wrt. y.
        """

        return self.METRICS[metric](y_true=y, y_pred=self.predict(X), sample_weight=sample_weight, **kwargs)