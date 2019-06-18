## Run Bayesian CCA model
import numpy as np
import math 
import BIBFA as BCCA
import matplotlib.pyplot as plt
import pickle

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

# Generate some data from the model, with pre-specified
# latent components
S = 2  #sources
Ntrain = Ntest = 100
N = Ntrain + Ntest
d = np.array([15, 7]) # dimensions
K = 4                 # components
Z = np.zeros((N, K))
j = 0
for i in range(0, N):
    Z[i,0] = np.sin((i+1)/(N/20))
    Z[i,1] = np.cos((i+1)/(N/20))
    if i < Ntrain:
        Z[i,3] = 2*((i+1)/Ntrain-0.5)
    else:
        Z[i,3] = 2*((j+1)/Ntest-0.5)
        j += 1        
Z[:,2] = np.random.normal(0, 1, N)

tau = np.array([3, 6])

#ARD parameters
alpha = np.zeros((S, K))
alpha[0,:] = np.array([1,1,1e8,1])
alpha[1,:] = np.array([1,1,1,1e8])

X = [[] for _ in range(d.size)]
X_train = [[] for _ in range(d.size)]
X_test = [[] for _ in range(d.size)]
W = [[] for _ in range(d.size)]
for i in range(0, d.size):
    W[i] = np.zeros((d[i], K))
    for k in range(0, K):
        W[i][:,k] = np.random.normal(0, 1/np.sqrt(alpha[i,k]), d[i])
    X[i] = (np.dot(Z,W[i].T) + np.reshape(
        np.random.normal(0, 1/np.sqrt(tau[i]), N*d[i]),(N, d[i])))
    X_train[i] = X[i][0:Ntrain,:]
    X_test[i] = X[i][Ntrain:N,:]

Z_train = Z[0:Ntrain,:]
Z_test = Z[Ntrain:N,:]  
X = X_train

 # Complete data
#------------------------------------------------------------------------
## Fitting the model and plotting weight means
m = 8 #number of models
BCCA = BCCA.BIBFA(X, m, d)
L = BCCA.fit(X)

with open('BCCA_para.dictionary', 'wb') as parameters:
 
  # Step 3
  pickle.dump(BCCA, parameters)

#Hinton diagrams for W1 and W2
W1 = BCCA.means_w[0]
W2 = BCCA.means_w[1]
W = np.concatenate((W1,W2),axis=0)
hinton(W)

#Hinton diagrams for alpha1 and alpha2
a1 = np.reshape(BCCA.E_alpha[0],(m,1))
a2 = np.reshape(BCCA.E_alpha[1],(m,1))
a = np.concatenate((a1,a2),axis=1)
hinton(-a.T)

print("Estimated variances:", BCCA.E_tau)
print("Estimated alphas:", BCCA.E_alpha)

#plot lower bound
plt.plot(L[2:])
plt.show()

#plot true latent variables
x = np.linspace(0,99,100)
f, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, sharex='col', sharey='row')
f.suptitle('True latent components')
ax1.scatter(x,Z_train[:,0])
ax2.scatter(x,Z_train[:,1])
ax3.scatter(x,Z_train[:,2])
ax4.scatter(x,Z_train[:,3])
plt.show()

#plot estimated latent variables
x = np.linspace(0,99,100)
f, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, sharex='col', sharey='row')
f.suptitle('Estimated latent components')
ax1.scatter(x,BCCA.means_z[:,1])
ax2.scatter(x,BCCA.means_z[:,4])
ax3.scatter(x,BCCA.means_z[:,5])
ax4.scatter(x,BCCA.means_z[:,6])
plt.show()

