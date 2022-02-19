from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class PCA(BaseEstimator):
    
    """Performs PCA on a given dataset. Assumes n_samples in
    X is greater than n_features.
    
    Parameters
    ----------
    n_components : {int, float}, default=None
        The number of principle components to keep. If none, keeps all.
        If 0 < n_components < 1, then the number of components kept is
        selected such that the fraction of total variance explained is
        at least n_components.
        
    Attributes
    ----------
    X_ : pd.DataFrame, shape (n_samples, n_features)
        The training samples passed during fit().
    y_ : pd.Series, shape (n_samples,)
        The labels passed during fit().
    is_fitted_: bool
        Boolean for whether fit() has been called on the current instance, used by 
        check_is_fitted().
    n_samples_: int
        Number of samples seen during fit().
    n_features_in_: int
        Number of features seen during fit().
    scaler_: sklearn.preprocessing._data.StandardScaler
        Scaler used to mean-center the data during fit().
    cov_mat_: ndarray
        Covariance matrix of X_
    explained_variance_: ndarray
        Eigvenvalues of X_, largest to smallest
    pcs_: ndarray
        Eigenvectors of X_ associated with explained_variance_, PCs
    explained_variance_ratio_: ndarray
        Eigenvalues as ratio of total variance
    """
    
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        
        
    def fit(self, X: pd.DataFrame, y=None):
        
        """Fit the estimator.
        
        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Samples.
            
        y: ignored
            
        Returns
        -------
        self : object
            Returns self to allow chaining function calls after fit().
        """
        
        # Boilerplate stuff
        
        X = check_array(X)
        
        self.is_fitted_ = True
        self.n_samples_ , self.n_features_in_ = X.shape
        self.X_ = X
        
        # validate n_components
        if self.n_components < 0 or self.n_components > self.X_.shape[1]:
            raise ValueError('n_components must be between 0 and n_features')
        
        # PCA performs a rotation of axes about the origin, so the data must be mean-centered
        self.scaler_ = StandardScaler(with_std=False)
        X = self.scaler_.fit_transform(self.X_)
        
        self.cov_mat_ = X.T.dot(X)/(self.n_samples_ - 1)
        
        # explained_variance_ and pcs_ contain the eigenvalues and vectors 
        # (as column vectors) of the covariance matrix
        self.explained_variance_, self.pcs_ = np.linalg.eig(self.cov_mat_)
        
        # sort by descending variance explained, i.e. explained_variance_[0] is the 
        # variance of the data along the first PC (components_[:, 0]), which is the PC 
        # with scores that have the highest variance
        order = np.argsort(self.explained_variance_)[::-1]
        self.explained_variance_ = self.explained_variance_[order]
        self.pcs_ = self.pcs_[:, order]
        self.explained_variance_ratio_ = self.explained_variance_/self.explained_variance_.sum()
        
        # select n_components if n_components is a fraction
        if self.n_components > 0 and self.n_components < 1:
            self.n_components = self.find_n_comps(self.n_components, self.explained_variance_ratio_)
        
        # keep only n_components PCs
        self.explained_variance_ = self.explained_variance_[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components]
        self.pcs_ = self.pcs_[:, :self.n_components]
    
        return self
    
    
    def find_n_comps(self, n_comp, explained_var):
        cumm = 0
        for i in range(len(explained_var)):
            cumm += explained_var[i]
            if cumm >= n_comp:
                break
        return i + 1
    
    
    def transform(self, X: pd.DataFrame):
        
        """Apply dim-red on X. Projects X onto the first PCs 
        found during fit(), reducing dimensionality to n_components.
        
        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Samples to reduce dimensionality.
            
        Returns
        -------
        X_new: pd.DataFrame, shape (n_samples, n_components)
            Projection of X onto the first principal components.
        """
        
        check_is_fitted(estimator=self, attributes='is_fitted_')
        X = pd.DataFrame(X)
        index = X.index
        X = self.scaler_.transform(X)  # converts to ndarray
        return pd.DataFrame(X.dot(self.pcs_), index=index)
    
    
    def inverse_transform(self, X: pd.DataFrame):
        """Transform X back into the original space. If the
        n_components dimension of X is less than n_features, 
        then the inverse transform will be an approximation
        of the original X.
        
        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_components)
            Samples in reduced dim.
            
        Returns
        -------
        X_orig: pd.DataFrame, shape (n_samples, n_features)
            Samples in original dim.
        """
        
        check_is_fitted(estimator=self, attributes='is_fitted_')
        X = pd.DataFrame(X)
        return pd.DataFrame(X.dot(self.pcs_.T) + self.scaler_.mean_, X.index)