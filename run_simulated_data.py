## Run Bayesian CCA model
import numpy as np
from scipy.linalg import block_diag, eig
import BayesianCCA
import matplotlib.pyplot as plt

#np.random.seed(42)

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
d = np.array([6, 8])
m = np.min(d)
SNR = 0
while SNR != 0.20:

    ## phi
    phi1 = np.random.randn(d[0], d[0])
    phi2 = np.random.randn(d[1], d[1])
    phi = block_diag(phi1,phi2)
    vals_phi, vecs_phi = eig(phi)
    maxval = np.max(np.real(vals_phi))

    # create W1 and W2
    W1 = np.random.randn(d[0], m)
    W2 = np.random.randn(d[1], m)
    W = np.concatenate((W1,W2),axis=0)
    vals_W, vecs_W = eig(np.dot(W.T,W))
    minval = np.min(np.real(vals_W))

    #Signal-to-noise ratio
    SNR = round(maxval/minval,2)

## Z
Z = np.random.normal(0, np.diag([10,8,1,1,1,1]))

# Drawing samples from N(x|mu,sigma)
sigma = np.dot(W, W.T) + phi
mu = np.dot(W,np.diagonal(Z))
N = 80
X = np.random.multivariate_normal(mu, sigma, N)

# Inputs for VCCA model
X = [X[:,0:d[0]],X[:,d[0]:d[0]+ d[1]]]
a = b = beta = np.array([10e-03, 10e-03])
K = np.array([10e-03 * np.identity(d[0]),10e-03 * np.identity(d[1])])
nu = np.array([d[0],d[1]])

# Fitting model and plotting weight means
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