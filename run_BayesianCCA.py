## Run Bayesian CCA model
import numpy as np
import BayesianCCA
import scipy as sp

np.random.seed(42)

## Simulated data
d = np.array([6,8])
m = np.min(d)
N = 80
phi1 = np.diag([4,3,1,1,1,1])
phi2 = np.diag([5,4,1,1,1,1])
W1

X1 = np.random.multivariate_normal(np.zeros(m), phi1, N)
X2 = np.random.multivariate_normal(np.zeros(m), phi2, N)

X = [X1,X2]
a = b = beta = np.array([10e-03, 10e-03])
K = np.array([10e-03 * np.identity(d[0]),10e-03 * np.identity(d[1])])
nu = np.array([1 + d[0],1 + d[1]])

BCCA = BayesianCCA.VCCA(d, N, a, b, beta, K, nu)
BCCA.fit(X) 

""" X1 = np.random.multivariate_normal(np.zeros(5), np.diag([3,2,1,1,1]), 50)
X2 = np.random.multivariate_normal(np.zeros(3), np.diag([2,1,1]), 50)

X = [X1,X2]
d = np.array([X1.shape[1], X2.shape[1]])
a = b = beta = np.array([10e-03, 10e-03])
N = X1.shape[0]
K = np.array([10e-03*np.identity(d[0]),10e-03*np.identity(d[1])])
nu = np.array([1 + d[0],1 + d[1]])

BCCA = BayesianCCA.VCCA(d, N, a, b, beta, K, nu)
BCCA.fit(X) """