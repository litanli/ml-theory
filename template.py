from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import numpy as np

class EstimatorName(BaseEstimator):
    
    """Short description
    
    Parameters
    ----------
    param1 : str, default='default'
        description
        
    param2 : str, default='default'
        description
        
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
    
    METRICS = {
        'accuracy_score': accuracy_score,
        'precision_score': precision_score,
        'recall_score': recall_score,
        'f1_score': f1_score
    }
    
    METRICS = {
        'rmse': mean_squared_error, 
        'mae': mean_absolute_error, 
        'mape': mean_absolute_percentage_error
    }

    
    def __init__(self, param1=None, param2=None):
        """Each constructor params should have a default value, allowing estimator 
        instantiation without passing any arguments. Params should be hyperparams 
        and should be documented under the "Parameters" section.
        """
        self.param1 = param1
        self.param2 = param2
        
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        
        """Fit the estimator. Attributes estimated from data should be suffixed with an 
        underscore, e.g. self.coeff_.
        
        Parameters
        ----------
        X : {pd.DataFrame, ndarray}, shape (n_samples, n_features)
            Training samples.
        y : {pd.Series, pd.DataFrame, ndarray}, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
            
        Returns
        -------
        self : object
            Returns self to allow chaining function calls after fit().
        """
        
        # Boilerplate stuff
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # X = check_array(X)
        X, y = pd.DataFrame(X), pd.Series(y)
    
        # Extract unique labels into an array
        self.classes_ = unique_labels(y)
        
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]
        
        self.X_ = X
        self.y_ = y
            
        return self
    
    
    def predict(self, X):
        
        """Predict class labels for X.
        
        Parameters
        ----------
        X : {pd.DataFrame, ndarray}, shape (n_samples, n_features)
            Samples to get predictions.
            
        Returns
        -------
        y : pd.Series, shape (n_samples,) or (n_samples, n_outputs)
            Predictions.
        """
        
        check_is_fitted(estimator=self, attributes='is_fitted_')
        
        X = pd.DataFrame(X)
        y = pd.Series(0, index=X.index)
        
        return y
    
    
    def predict_proba(self, X):
        
        """Predict class proba for samples in X.
        
        Parameters
        ----------
        X : {pd.DataFrame, ndarray}, shape (n_samples, n_features)
            Samples to get class proba.
            
        Returns
        -------
        y : pd.Series, shape (n_samples,) or (n_samples, n_outputs)
            Predicted probabilities.
        """
        
        check_is_fitted(estimator=self, attributes='is_fitted_')
        
        X = pd.DataFrame(X)
        y = pd.Series(0, index=X.index)
        
        return y
    
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