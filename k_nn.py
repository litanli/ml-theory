import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import sys

def knn(X_train, y_train, X_pred, k, metric = "minkowski", p = 2):
    """
    Performs k-nearest neighbors classification using majority vote. Note in 
    the knn algorithm, no internal model is built. This is unlike in algorithms 
    such as neural networks or logistic regression, where a set of model 
    parameters are calculated. The heuristic chosen for tie-breaking is to pick 
    the class of the closest neighbor amongst the tied neighbors. E.g. k=5, the 
    classes of the 5 nearest neighbors for a certain new example are (in order 
    of closest to farthest): class 0, class 2, class 1, class 0, and class 2. 
    There's a tie between class 0 and 2, so we choose the class of the closest 
    neighbor, 0, as the predicted class.
    
    
    Arguments:
    X_train -- training set represented by numpy array of shape (m_train, n)
    
    y_train -- training labels represented by numpy array (vector) of shape 
               (m_train, 1). Values should be integers, representing classes
               
    X_pred -- set of unlabeled observations to make predictions on, 
              represented by numpy array of shape (m_pred, n)
              
    k -- number of nearest neighbors
    
    metric -- the distance metric used for finding nearest neighbors. Defaults 
              to minkowski with p = 2 (same as euclidean). Possible values:
              "minkowski" - minkowski distance. Default order set to p = 2,
                            which is the euclidean distance. Set p = 1 for 
                            Manhattan distance
              "sorensen"  - sorensen distance
              
    p -- optional parameter for order of minkowski distance metric
    
    Returns:
    y_pred -- predictions made on X_pred, represented by numpy array (vector) 
              of shape (m_pred, 1)
              
    knn_row_idx -- Row indices of the X_train matrix for the k nearest 
                   neighbors found for each prediction example, represented by 
                   numpy array of shape (m_pred, k). 
    """
    
    # make sure y_train is a numpy array, and of shape (m_train, 1) instead of
    # (m_train, )
    try:
        y_train.shape[1]
    except IndexError:
        print("y_train must be a numpy array of shape (m_train, 1), not (m_train, )")
        sys.exit(1)
    except AttributeError:
        print("y_train must be a numpy array")
        sys.exit(1)
    
    # make sure 0 < k <= m_train
    if k <= 0 or k > X_train.shape[0]:
        print("Error: k must be 0 < k <= m_train")
        sys.exit(1)
    
    # make sure metric is of string type, and of a valid value
    allowed = ["minkowski", "sorensen"]
    if not isinstance(metric, str):
        print("Error: metric parameter must be a string")
        sys.exit(1)
    elif metric not in allowed:
        print("Error: metric must be of an allowed type: \"minkowski\", \"sorensen\", etc.")
        sys.exit(1)
    else:
        pass
    
    
    # Go through each prediction example, find the k-nearest neighbors and the
    # majority vote for the label
    m_pred = X_pred.shape[0]
    y_pred = np.empty((m_pred, 1)); y_pred.fill(np.nan)
    knn_row_idx = np.empty((m_pred, k)); knn_row_idx.fill(np.nan)
    
    for i in range(m_pred):
        
        D = distance(X_train, X_pred[i, :], metric, p)
 
        nn_idx = np.argsort(D, axis = 0)
        knn_labels = y_train[nn_idx, :][0:k, :] # k x 1
        
        majority_voted_label = stats.mode(knn_labels).mode[0][0]

        # If tie, assign label of closest neighbor
        if stats.mode(knn_labels).count[0][0] == 1:
            majority_voted_label = knn_labels[0][0]
        
        y_pred[i] = majority_voted_label
        knn_row_idx[i, :] = nn_idx[0:k]
        
    return y_pred, knn_row_idx.astype(int)


def minkowski(X_train, x_pred, p):
    """
    Helper function. Returns the Minkowski distances (of order p) between a 
    single example x_pred and each example in X_train.
    
    Arguments:
    X_train -- training set represented by numpy array of shape (m_train, n)
    x_pred -- new example represented by numpy array of shape (1, n)
    
    Returns:
    D -- distances between x_pred and each example in X_train, represented by
         numpy array of shape (m_train,)
    """
    m_train = X_train.shape[0]
    # x_pred is broadcasted into shape (m_train, n) before subtracting
    D = np.power(np.sum(np.abs(X_train - x_pred)**p, axis = 1), 1/p) 
    
    return D

def sorensen(X_train, x_pred):
    """
    Helper function. Returns the Sorensen distances between a single example 
    x_pred and each example in X_train.
    
    Arguments:
    X_train -- training set represented by numpy array of shape (m_train, n)
    x_pred -- new example represented by numpy array of shape (1, n)
    
    Returns:
    D -- distances between x_pred and each example in X_train, represented by
         numpy array of shape (m_train,)
    """
    m_train = X_train.shape[0]
    # x_pred is broadcasted into shape (m_train, n) before subtracting
    D = np.sum(np.abs(X_train - x_pred), axis = 1) / np.sum(X_train + x_pred, axis = 1)
    
    return D

def distance(X_train, x_pred, metric, p):
    switcher = {
            "minkowski": minkowski(X_train, x_pred, p),
            "sorensen": sorensen(X_train, x_pred)
    }
    return switcher.get(metric)


#----------------- Test knn by comparing to Scikit-learn ---------------------#
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values # Age, Estimated Salary
y = dataset.iloc[:, 4].values # Purchased

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_S = sc.fit_transform(X_train)
X_test_S = sc.transform(X_test)

# ------------------------- Scikit-learn results -----------------------------#
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train_S, y_train)

# Predicting the Test set results
y_pred_skl = classifier.predict(X_test_S)
y_pred_skl = np.array([y_pred_skl]).T

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_skl = confusion_matrix(y_test, y_pred_skl)

# ------------------------- kNearestNeighbors results ------------------------#
y_train = np.array([y_train]).T # change y_train's shape from (300,) to (300, 1)
y_pred, knn_row_idx = knn(X_train_S, y_train, X_test_S, k = 5, metric = "minkowski", p = 2)
assert(np.array_equal(y_pred_skl, y_pred)) # Good!
cm = confusion_matrix(y_test, y_pred)
assert(np.array_equal(cm_skl, cm)) # Good!
