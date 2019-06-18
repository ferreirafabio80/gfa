## Run Bayesian CCA model
import numpy as np
import math 
import BIBFA_missing as BCCA
import matplotlib.pyplot as plt
import pickle

np.random.seed(42)

# Generate some data from the model, with pre-specified
# latent components

S = 2  #sources
Ntrain = Ntest = 200
N = Ntrain + Ntest
d = np.array([10, 7]) # dimensions
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

#Diagonal noise precisions
tau = [[] for _ in range(d.size)]
tau[0] = np.array([7**2, 6**2, 5**2, 4**2, 1, 1, 1, 1, 1, 1])
tau[1] = np.array([12**2, 11**2, 12**2, 8**2, 1, 1, 1])

#ARD parameters
alpha = np.zeros((S, K))
alpha[0,:] = np.array([1,1,1e10,1])
alpha[1,:] = np.array([1,1,1,1e10])

X = [[] for _ in range(d.size)]
X_train = [[] for _ in range(d.size)]
X_test = [[] for _ in range(d.size)]
W = [[] for _ in range(d.size)]
for i in range(0, d.size):
    W[i] = np.zeros((d[i], K))
    for k in range(0, K):
        W[i][:,k] = np.random.normal(0, 1/np.sqrt(alpha[i,k]), d[i])
    
    X[i] = np.dot(Z, W[i].T) + np.random.multivariate_normal(
            np.zeros((1, d[i]))[0], np.diag(1/np.sqrt(tau[i])), N)
    X_train[i] = X[i][0:Ntrain,:]
    X_test[i] = X[i][Ntrain:N,:]

Z_train = Z[0:Ntrain,:]
Z_test = Z[Ntrain:N,:]  
X = X_train

# Incomplete data
#------------------------------------------------------------------------
p_miss = 0.10
for i in range(0,2):
    missing =  np.random.choice([0, 1], size=(200,d[i]), p=[1-p_miss, p_miss])
    X[i][missing == 1] = 'NaN'

m  = 8 #number of models
BCCA = BCCA.BIBFA(X, m, d)
L = BCCA.fit(X)
BCCA.L = L

with open('BCCAdiag_missing10_sample200.dictionary', 'wb') as parameters:
 
  # Step 3
  pickle.dump(BCCA, parameters)

#plot true latent variables
""" x = np.linspace(0,99,100)
f, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, sharex='col', sharey='row')
f.suptitle('True latent components')
ax1.scatter(x,Z_train[:,0])
ax2.scatter(x,Z_train[:,1])
ax3.scatter(x,Z_train[:,2])
ax4.scatter(x,Z_train[:,3])
plt.show() """