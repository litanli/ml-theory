import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sym
from sympy.matrices import Matrix

#data preprocessing
#x1 = np.array([1, 3])
#x2 = np.array([1, 2])
x1 = np.array([1, 2, 2.9, 2, 4.5])
x2 = np.array([1, -1, 2, 4, 0])
X = np.column_stack((x1,x2))
y = np.array([-1,1])
y = np.array([-1,-1,1,1,1])

N = y.size
p = X.shape[1]

#plot data
x1_neg = np.array([])
x2_neg = np.array([])
x1_pos = np.array([])
x2_pos = np.array([])
for i in range(0, N):
    if y[i]==-1:
        x1_neg = np.append(x1_neg,x1[i])
        x2_neg = np.append(x2_neg,x2[i])
    else:
        x1_pos = np.append(x1_pos,x1[i])
        x2_pos = np.append(x2_pos,x2[i])

plt.close("all")
fig, ax = plt.subplots()
ax.scatter(x1_neg, x2_neg, marker = "_", label = "negative observations")
ax.scatter(x1_pos, x2_pos, marker = "+", label = "positive observations")
ax.legend()
plt.title("Data")
plt.xlabel("x1")
plt.ylabel("x2")

# steps
# find λ's
#λ1, λ2 = sym.symbols('λ1:3', real=True) # modify this depending on value of N. Number of λ's = N
λ1, λ2, λ3, λ4, λ5 = sym.symbols('λ1:6', real=True) # modify this depending on value of N. Number of λ's = N
#λ = np.array([λ1, λ2]).T # modify this depending on value of N
λ = np.array([λ1, λ2, λ3, λ4, λ5]).T # modify this depending on value of N

D = np.empty((N,N))
for i in range(0, N):
    for j in range(0, N):
        D[i,j]=y[i]*y[j]*X[i,:].dot(X[j,:])
        
# Σλiyi = 0
# sub out λ_N
num = 0
for i in range(0, N-1):
    num -= λ[i]*y[i]
λ[N-1] = num/y[N-1] 
       
obj_func = λ.T.dot(np.ones(N).T)-1/2*λ.T.dot(D).dot(λ)

sys_of_eqns = []
for i in range(0, N-1):
    sys_of_eqns.append(sym.diff(obj_func, λ[i])) 
    
    
    
G = np.array([[-13.25,-7.75,7.6,12.75],[-7.75,-7.25,2,2.25],[7.6,2,-6.56,-12],[12.75,2.25,-12,-22.25]])
H = np.array([2,2,0,0])
np.linalg.inv(G).dot(H)

sys_of_eqns = tuple(sys_of_eqns)
λ = list(sym.solve(sys_of_eqns, λ[0]).values()) # modify this depending on value of N. Solve for all λ's except for λ_N

num = 0
for i in range(0, N-1):
    num -= λ[i]*y[i]
λ2 = num/y[N-1] # solve for λ_N. Modify depending on N 
λ.append(λ2)

#check that lambdas are all positive to fulfill KKT
for i in range(0, N):
    if λ[i] < 0:
        raise ValueError("λ" + str(i+1) + " is negative.")

# find w
w = np.sum([float(λ[i])*y[i]*X[i,:] for i in range(0, N)], axis = 0)
w_hat = w/np.linalg.norm(w)
#plt.plot([0,w[0]],[0,w[1]], label="w-dir, mine")

# find d_om
for i in range(0, N):
    if λ[i] > 0 & (y[i] == 1 | y[i] == -1):
        d_om = X[i,:].dot(w)/np.linalg.norm(w)-y[i]/np.linalg.norm(w)
        break

# plot street - modify for p > 2
# middle of street
start = d_om*w_hat
plt.plot([start[0], start[0]+5*w_hat[1]],[start[1], start[1]+5*(-w_hat[0])], c = "blue") 
plt.plot([start[0], start[0]-5*w_hat[1]],[start[1], start[1]-5*(-w_hat[0])], c = "blue") 
# margin
start = (d_om+1/np.linalg.norm(w))*w_hat
plt.plot([start[0], start[0]+5*w_hat[1]],[start[1], start[1]+5*(-w_hat[0])], c = "red") 
plt.plot([start[0], start[0]-5*w_hat[1]],[start[1], start[1]-5*(-w_hat[0])], c = "red") 
# other margin
start = (d_om-1/np.linalg.norm(w))*w_hat
plt.plot([start[0], start[0]+5*w_hat[1]],[start[1], start[1]+5*(-w_hat[0])], c = "red") 
plt.plot([start[0], start[0]-5*w_hat[1]],[start[1], start[1]-5*(-w_hat[0])], c = "red") 





import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm

x1 = np.array([1, 2, 2.9, 2, 4.5])
x2 = np.array([1, -1, 2, 4, 0])
plt.scatter(x1,x2)
plt.show()
X = np.column_stack((x1,x2))
y = [0, 0, 1, 1, 1]


#define classifier
clf = svm.SVC(kernel='linear', C = 1)
clf.fit(X,y)

#predicting new observations
print(clf.predict([0.58,0.76]))
print(clf.predict([10.58,10.76]))

#visalize
w = clf.coef_[0]
print(w)

a = -w[0] / w[1]

xx = np.linspace(0,12)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

plt.scatter(X[:, 0], X[:, 1], c = y)
plt.ylim(-5,5)
plt.xlim(-5,5)
plt.legend()
plt.show()



    









