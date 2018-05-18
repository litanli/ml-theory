# PCA - can be used to reduce the number of variables of a dataset to make some 
# machine learning algorithms faster.

# perform PCA dimensionality reduction and compare to sklearn package results
# to verify accuracy
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X1 = iris.data
species = {0:"Iris-setosa", 1:"Iris-versicolor", 2:"Iris-virginica"}
y = [species[num] for num in iris.target]

# standardize independent variables
X1_mean = np.zeros(shape=(X1.shape[0],X1.shape[1]))
X1_stdev = np.zeros(shape=(X1.shape[0],X1.shape[1]))
N = X1.shape[0]

for j in range(0, X1.shape[1]):
    sums = 0
    for i in range(0, X1.shape[0]):
        sums += X1[i][j]
    for i in range(0, X1.shape[0]):
        X1_mean[i][j] = sums/N

# assume data is from a random sample; use N divisor
for j in range(0, X1.shape[1]):
    sum_of_sq_residuals = 0
    for i in range(0, X1.shape[0]):
        sum_of_sq_residuals += (X1[i][j]-X1_mean[i][j])**2
    print(sum_of_sq_residuals)
    for i in range(0, X1.shape[0]):
        X1_stdev[i][j] = (sum_of_sq_residuals/N)**0.5

X1 = (X1-X1_mean)/X1_stdev  

# correlation matrix
S = 1/(N-1)*X1.transpose().dot(X1) 

# We let the PCs be linear combinations of all the IVs, but with loadings that 
# give max variance (correlates well with the data points), and uncorrelated 
# with all other PCs. 

# More formally:
# the ith principle component t_i is the linear combination of maximum variance 
# that is uncorrelated with all previous PCs (axes orthogonal to all previous 
# new axes), with loadings for each PC represented in vectors u_i
#	Find u_i ∋ u_i^T*S*u_i is maximum
#	subject to constraint ‖u_i‖=u_i'*u_i=1 for a unique solution
#	and u_k'*S*u_i=0 ∀ k < i

# Using Lagrange multiplier λ the optimization problem reduces to an eigenvalue
# problem of the form S*u_i = λ_i*u_i.

eigenvalues, eigenvectors = np.linalg.eig(S)

# sort eigenvalues from largest to smallest (and corresponding eigenvectors)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]

# var(t_i) = λ_i. Intuitively, the larger the variance of a PC, the more that 
# PC correlates with the data. Therefore, a simple measure of how well each PC
# correlates with the data is λ_i/Σ(λ_i).
 
# We choose PC1 and PC2, reducing dimension to l = 2 yet explaining over 95% 
# of variability in the data! 

(eigenvalues[0]+eigenvalues[1])/np.sum(eigenvalues)
u1 = eigenvectors[:,0]
u2 = -eigenvectors[:,1]

# regression on data
t1 = X1.dot(u1)
t2 = X1.dot(u2)
T = np.column_stack((t1,t2))
X1_hat = T.dot(np.linalg.inv(T.transpose().dot(T))).dot(T.transpose()).dot(X1)
residuals = X1-X1_hat # conclusion: we're able to represent the data using just
# two variables (PCs), and the data's pretty close to the original.

# plot the data now represented by two PCs
principalDf = pd.DataFrame(data = T, columns = ['Principal Component 1',
                                                  'Principal Component 2'])
speciesDf = pd.DataFrame(data = y, columns = ['Species'])
finalDf = pd.concat([principalDf, speciesDf], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2-Component PCA', fontsize = 20)
species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for specie, color in zip(species,colors):
    indicesToKeep = finalDf['Species'] == specie
    ax.scatter(finalDf.loc[indicesToKeep, 'Principal Component 1']
               , finalDf.loc[indicesToKeep, 'Principal Component 2']
               , c = color
               , s = 50)
ax.legend(species)
ax.grid()


# compare to sklearn package results to verify accuracy
from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target
species = {0:"Iris-setosa", 1:"Iris-versicolor", 2:"Iris-virginica"}
y = [species[num] for num in iris.target]

X = StandardScaler().fit_transform(X) 

pca = decomposition.PCA(n_components = 2)
PCs = pca.fit_transform(X) 

principalDf = pd.DataFrame(data = PCs, columns = ['Principal Component 1',
                                                  'Principal Component 2'])
speciesDf = pd.DataFrame(data = y, columns = ['Species'])
finalDf = pd.concat([principalDf, speciesDf], axis = 1)

# plot
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2-Component PCA', fontsize = 20)
species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for specie, color in zip(species,colors):
    indicesToKeep = finalDf['Species'] == specie
    ax.scatter(finalDf.loc[indicesToKeep, 'Principal Component 1']
               , finalDf.loc[indicesToKeep, 'Principal Component 2']
               , c = color
               , s = 50)
ax.legend(species)
ax.grid()