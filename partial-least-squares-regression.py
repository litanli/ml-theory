# PLS - Multiple linear regression will not work when there's multicollinearity
# (then the OLS solution to multiple linear regression does not exist since X'X
# is not invertible) or if the data matrix X is ill-conditioned. PLS can be 
# used instead, projecting X and Y data into a new space. Its idea is similar
# to principle component analysis, except that the objective function considers
# the responses as well as the predictors, seeking the lowest residuals for 
# both.
 
# perform PLS regression and compare to sklearn package results to verify 
# accuracy

import numpy as np
np.set_printoptions(threshold=np.inf)
from sklearn import datasets
import matplotlib.pyplot as plt

# Multiple linear regression
# y = np.array([41, 49, 69, 65, 40, 50, 58, 57, 31, 36, 44, 57, 19, 31, 33, 43])
# x1 = np.array([1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4])
# x2 = np.array([5,5,5,5,10,10,10,10,15,15,15,15,20,20,20,20])
# x0 = np.ones(x1.size)
# X = np.column_stack((x0, x1, x2))
# b_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
# y_hat = X.dot(b_hat)
# residuals = y - y_hat

# PLS
# data preprocessing
#y = np.array([41, 49, 69, 65, 40, 50, 58, 57, 31, 36, 44, 57, 19, 31, 33, 43])
#x1 = np.array([1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4])
#x2 = np.array([5,5,5,5,10,10,10,10,15,15,15,15,20,20,20,20])
#x3 = 2*x2
#X = np.column_stack((x1, x2, x3))

X_mean = np.zeros(shape=(X.shape[0],X.shape[1]))
X_stdev = np.zeros(shape=(X.shape[0],X.shape[1]))
y_mean = np.zeros(shape=(y.size))
y_stdev = np.zeros(shape=(y.size))
N = X.shape[0]
m = X.shape[1]

for j in range(0, X.shape[1]):
    sums = 0
    for i in range(0, X.shape[0]):
        sums += X[i][j]
    for i in range(0, X.shape[0]):
        X_mean[i][j] = sums/N

sums = 0
for i in range(0, y.shape[0]):
    sums += y[i]
for i in range(0, y.shape[0]):
    y_mean[i] = sums/N

# assume data is from population, N divisor
for j in range(0, X.shape[1]):
    sum_of_sq_residuals = 0
    for i in range(0, X.shape[0]):
        sum_of_sq_residuals += (X[i][j]-X_mean[i][j])**2
    for i in range(0, X.shape[0]):
        X_stdev[i][j] = (sum_of_sq_residuals/N)**0.5

sum_of_sq_residuals = 0
for i in range(0, y.shape[0]):
    sum_of_sq_residuals += (y[i]-y_mean[i])**2
for i in range(0, y.shape[0]):
    y_stdev[i] = (sum_of_sq_residuals/N)**0.5

X = (X-X_mean)/X_stdev
y = (y-y_mean)/y_stdev

#steps
#find T and U
U = np.zeros([m, np.linalg.matrix_rank(X)])
T = np.zeros([N, np.linalg.matrix_rank(X)])
P = np.zeros([m, np.linalg.matrix_rank(X)])
Q = np.zeros(np.linalg.matrix_rank(X))
i = 0
Xi = X
yi = y
while (np.linalg.norm(Xi) > 1E-6): #Frobenius norm
    num = 0;
    for k in range(0, N):
        num += yi[k]*Xi[k,:]
    U[:,i] = num/np.linalg.norm(num)
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
epsilon_x = X

while (np.linalg.norm(epsilon_x)/np.linalg.norm(X) > 1-min_variance_explained): #Frobenius norm
    print(np.linalg.norm(epsilon_x)/np.linalg.norm(X))
    epsilon_x = epsilon_x - np.outer(T[:,l-1], P[:,l-1])
    l += 1

l = l-1 #final answer for l, to account for the last l += 1

#solving for y_hat
T_final = np.zeros([N, l])
for i in range(0, l):
    T_final[:,i] = T[:,i]


alpha_hat = np.linalg.inv(T_final.transpose().dot(T_final)).dot(T_final.transpose()).dot(y)
y_pred = T_final.dot(alpha_hat)
residuals = y - y_pred

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

X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
Y = [0.1, 0.9, 6.2, 11.9]
pls2 = PLSRegression(n_components=3)
pls2.fit(X, Y)
Y_pred = pls2.predict(X)




y = np.array([41, 49, 69, 65, 40, 50, 58, 57, 31, 36, 44, 57, 19, 31, 33, 43])
x1 = np.array([1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4])
x2 = np.array([5,5,5,5,10,10,10,10,15,15,15,15,20,20,20,20])
x3 = 2*x2
X = np.column_stack((x1, x2, x3))

X = StandardScaler().fit_transform(X) # N divisor in stdev.
y = StandardScaler().fit_transform(y) # N divisor in stdev.

pls = PLSRegression(n_components = 2)
pls.fit_transform(X,y) 
y_pred = pls2.predict(X)