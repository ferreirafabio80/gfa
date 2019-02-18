## Run Bayesian CCA model
import numpy as np
import BayesianCCA

X1 = np.random.multivariate_normal(np.zeros(10), np.diag([5,4,3,2,1,1,1,1,1,1]), 100).T
X2 = np.random.multivariate_normal(np.zeros(10), np.diag([5,4,3,2,1,1,1,1,1,1]), 100).T

X = [X1,X2]
d = np.array([X1.shape[1], X2.shape[1]])
a = b = beta = np.array([1e-03, 1e-03])
N = np.array([X1.shape[0], X2.shape[0]])
K = np.array([1e-03*np.identity(d[0]),1e-03*np.identity(d[1])])
nu = np.array([1 + d[0],1 + d[1]])

BCCA = BayesianCCA.VCCA(d, N, a, b, beta, K, nu)