# PLS - Multiple linear regression will not work when there's multicollinearity
# (then the OLS solution to multiple linear regression does not exist since X'X
# is not invertible) or if the data matrix X is ill-conditioned. PLS can be 
# used instead, projecting X and Y data into a new space. Its idea is similar
# to principle component analysis, except that the objective function considers
# the responses as well as the predictors
 
# perform PLS regression and compare to sklearn package results to verify 
# accuracy

import numpy as np
np.set_printoptions(threshold=np.inf)
from sklearn import datasets
import matplotlib.pyplot as plt

# data preprocessing
y = np.array([41, 49, 69, 65])
x1 = np.array([1,2,3,4])
x2 = np.array([5,4,4,5])
x3 = 2*x2
X = np.column_stack((x1, x2, x3))

X_mean = np.zeros(shape=(X.shape[0],X.shape[1]))
X_stdev = np.zeros(shape=(X.shape[0],X.shape[1]))
y_mean = np.zeros(shape=(y.size))
y_stdev = np.zeros(shape=(y.size))
N = X.shape[0]
m = X.shape[1]

for j in range(0, m):
    sums = 0
    for i in range(0, N):
        sums += X[i][j]
    for i in range(0, N):
        X_mean[i][j] = sums/N

sums = 0
for i in range(0, y.size):
    sums += y[i]
for i in range(0, y.size):
    y_mean[i] = sums/N

# use population stdev
for j in range(0, m):
    sum_of_sq_residuals = 0
    for i in range(0, N):
        sum_of_sq_residuals += (X[i][j]-X_mean[i][j])**2
    for i in range(0, N):
        X_stdev[i][j] = (sum_of_sq_residuals/N)**0.5
    
sum_of_sq_residuals = 0
for i in range(0, y.size):
    sum_of_sq_residuals += (y[i]-y_mean[i])**2
for i in range(0, y.size):
    y_stdev[i] = (sum_of_sq_residuals/N)**0.5

# standardize X and y
X_s = (X-X_mean)/X_stdev
y_s = (y-y_mean)/y_stdev

#find T and U
U = np.zeros([m, np.linalg.matrix_rank(X_s)])
T = np.zeros([N, np.linalg.matrix_rank(X_s)])
P = np.zeros([m, np.linalg.matrix_rank(X_s)])
Q = np.zeros(np.linalg.matrix_rank(X_s))
i = 0
Xi = X_s
yi = y_s
while (np.linalg.norm(Xi) > 1E-6): #Frobenius norm
    num = 0;
    for k in range(0, N):
        num += yi[k]*Xi[k,:]
    U[:,i] = -num/np.linalg.norm(num)
    T[:,i] = Xi.dot(U[:,i])
    P[:,i] = Xi.T.dot(T[:,i])/(T[:,i].T.dot(T[:,i]))
    Q[i] = T[:,i].T.dot(yi)/(T[:,i].T.dot(T[:,i]))

    Xi = Xi-np.outer(T[:,i], P[:,i]) #X_i+1
    yi = yi-T[:,i]*Q[i] #y_i+1
    i += 1
    print("i = ", i)
    print(np.linalg.norm(Xi))
print("Verify that Xi is close to zero.")
print(Xi)

# determine l, the number of latent vectors needed to explain 
# min_variance_explained fraction of the variance of the data
min_variance_explained = 0.85 # change this to suite your need
l = 1
epsilon_x = X_s

while (np.linalg.norm(epsilon_x)/np.linalg.norm(X_s) > 1-min_variance_explained): #Frobenius norm
    print(np.linalg.norm(epsilon_x)/np.linalg.norm(X_s))
    epsilon_x = epsilon_x - np.outer(T[:,l-1], P[:,l-1])
    l += 1

l = l-1 #final answer for l, to account for the last l += 1

#solving for y_hat
T_final = np.zeros([N, l])
for i in range(0, l):
    T_final[:,i] = T[:,i]


alpha_hat = np.linalg.inv(T_final.transpose().dot(T_final)).dot(T_final.transpose()).dot(y)
y_pred_s = T_final.dot(alpha_hat)
y_pred = y_pred_s*y_stdev+y_mean


#Scree plot
cummulative_variance_explained = np.zeros(np.linalg.matrix_rank(X))
epsilon_x = X
for i in range(0, np.linalg.matrix_rank(X)):
    for j in range(0, i+1):
        epsilon_x = epsilon_x - np.outer(T[:,j], P[:,j])
    cummulative_variance_explained[i] = 1-np.linalg.norm(epsilon_x)/np.linalg.norm(X)
    epsilon_x = X
x_axis = np.arange(1, np.linalg.matrix_rank(X)+1)
plt.scatter(x_axis, cummulative_variance_explained)
plt.plot(x_axis, cummulative_variance_explained)
plt.title("Scree Plot")
plt.xlabel("Number of latent vectors used")
plt.ylabel("Percentage of variance explained")
plt.xticks(x_axis, x_axis)
plt.yticks()
plt.show()

# compare to sklearn package results to verify accuracy
import numpy as np
np.set_printoptions(threshold=np.inf)
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

X = [[1,5,10],[2,4,8],[3,4,8],[4,5,10]]
y = [41, 49, 69, 65]

X = StandardScaler().fit_transform(X) # population stdev
y = StandardScaler().fit_transform(y) # population stdev

pls1 = PLSRegression(n_components=2)
scores = pls1.fit_transform(X, y)
T = pls1.x_scores_
W = pls1.x_weights_
P = pls1.y_loadings_

y_pred = pls1.predict(X)