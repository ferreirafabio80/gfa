## Run Bayesian CCA model
import numpy as np
import math 
import BayesianCCA_gamma as BCCA_gamma
import matplotlib.pyplot as plt

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

# Generate some data from the model, with pre-specified
# latent components
np.random.seed(42)
S = 2  #sources
Ntrain = Ntest = 100
N = Ntrain + Ntest
d = np.array([15, 7]) # dimensions
K = 4                 # components
Z = np.zeros((N, K))
j = 0
for i in range(0, N):
    Z[i,0] = math.sin(i+1/(N/20))
    Z[i,1] = math.cos(i+1/(N/20))
    if i < Ntrain:
        Z[i,3] = 2*((i+1)/Ntrain-0.5)
    else:
        Z[i,3] = 2*((j+1)/Ntest-0.5)
        j += 1        
Z[:,2] = np.random.normal(0, 1, N)

#Diagonal noise precisions
#phi = [[] for _ in range(d.size)]
#phi[0] = np.diag([7, 6, 5, 4, 2, 1, 1, 1])
#phi[1] = np.diag([10, 8, 5, 4, 1, 1])
tau = np.array([3, 6])

#ARD parameters
alpha = np.zeros((S, K))
alpha[0,:] = np.array([1,1,1e6,1])
alpha[1,:] = np.array([1,1,1,1e6])

X = [[] for _ in range(d.size)]
X_train = [[] for _ in range(d.size)]
X_test = [[] for _ in range(d.size)]
W = [[] for _ in range(d.size)]
for i in range(0, d.size):
    W[i] = np.zeros((d[i], K))
    for k in range(0, K):
        W[i][:,k] = np.random.normal(0, 1/np.sqrt(alpha[i,k]), d[i])
    X[i] = (np.dot(Z,W[i].T) + np.reshape(
        np.random.normal(0, np.sqrt(tau[i]), N*d[i]),(N, d[i])))
    X_train[i] = X[i][0:Ntrain,:]
    X_test[i] = X[i][Ntrain:N,:]

Z_train = Z[0:Ntrain,:]
Z_test = Z[Ntrain:N,:]  

## Fitting the model and plotting weight means
m = 8 #number of models
BCCA = BCCA_gamma.VCCA(X, m, d)
L = BCCA.fit(X)