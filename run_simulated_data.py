## Run Bayesian CCA model
import numpy as np
from scipy import linalg as LA
import BayesianCCA
import matplotlib.pyplot as plt

np.random.seed(42)

def hinton(matrix, max_weight=None, ax=None):
    #Draw Hinton diagram for visualizing a weight matrix.
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()
    plt.show()

## Simulated data
d = np.array([6,8])
m = np.min(d)
N = 80

## phi
phi1 = np.diag([4,3,1,1,1,1])
phi2 = np.diag([5,4,1,1,1,1,1,1])

# create W1 and W2
A1 = LA.orth(np.random.randn(d[0], 2))
A2 = LA.orth(np.random.randn(d[1], 2))
W1 = np.concatenate((A1, np.zeros((d[0],4))),axis=1)
W2 = np.concatenate((A2, np.zeros((d[1],4))),axis=1)

# Gaussian noise 
noise1 = 0.2 * np.random.normal(0,1,phi1.shape)
noise2 = 0.2 * np.random.normal(0,1,phi2.shape)
vals1, vecs1 = LA.eig(np.dot(W1.T,W1))
vals2, vecs2 = LA.eig(np.dot(W2.T,W2))
minval1 = np.min(np.real(vals1))
minval2 = np.min(np.real(vals2)) 

sigma1 = np.dot(W1, W1.T) + phi1 + noise1
sigma2 = np.dot(W2, W2.T) + phi2 + noise2

X1 = np.random.multivariate_normal(np.zeros(d[0]), sigma1, N)
X2 = np.random.multivariate_normal(np.zeros(d[1]), sigma2, N)

X = [X1,X2]
a = b = beta = np.array([10e-03, 10e-03])
K = np.array([10e-03 * np.identity(d[0]),10e-03 * np.identity(d[1])])
nu = np.array([1 + d[0],1 + d[1]])

BCCA = BayesianCCA.VCCA(d, N, a, b, beta, K, nu)
L = BCCA.fit(X)
hinton(BCCA.means_w[1])

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