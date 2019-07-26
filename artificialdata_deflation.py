# Run Bayesian CCA model
import numpy as np
import math
import GFA
import matplotlib.pyplot as plt
import pickle
import os
import time

data = 'deflation'
noise = 'PCA'
model = 'GFA'
m = 25  # number of models
N = 2000
directory = f'results/{data}/{m}models_{N}samples/'
if not os.path.exists(directory):
        os.makedirs(directory)

# Generate some data from the model, with pre-specified
# latent components
S = 2  # sources
d = np.array([60000, 300])  # dimensions
K = 4                 # components
Z = np.zeros((N, K))
j = 0

for i in range(0, N):
    Z[i, 0] = np.sin((i+1)/(N/20))
    Z[i, 1] = np.cos((i+1)/(N/20))
    Z[i, 2] = 2*((i+1)/N/2-0.5)
Z[:, 3] = np.random.normal(0, 1, N)

# spherical precisions
tau = np.array([3, 6])

# ARD parameters
alpha = np.zeros((S, K))
alpha[0, :] = np.array([1, 1, 1e8, 1])
alpha[1, :] = np.array([1, 1, 1, 1e8])

X = [[] for _ in range(d.size)]
W = [[] for _ in range(d.size)]
for i in range(0, d.size):
    W[i] = np.zeros((d[i], K))
    for k in range(0, K):
        W[i][:, k] = np.random.normal(0, 1/np.sqrt(alpha[i, k]), d[i])

    X[i] = (np.dot(Z, W[i].T) + np.reshape(
        np.random.normal(0, 1/np.sqrt(tau[i]), N*d[i]), (N, d[i])))
    arraypath = f'{directory}/X{i+1}.txt'
    #if not os.path.exists(arraypath):
    np.savetxt(arraypath, X[i])   


num_init = 1  # number of random initializations
res_BIBFA = [[] for _ in range(num_init)]
for init in range(0, num_init):
    print("Iterations:", init+1)

    # Complete data
    # ------------------------------------------------------------------------
    time_start = time.process_time()
    res_BIBFA[init] = GFA.BIBFA(X, m, d)
    L = res_BIBFA[init].fit(X)
    res_BIBFA[init].L = L
    res_BIBFA[init].Z = Z
    res_BIBFA[init].time_elapsed = (time.process_time() - time_start)

filepath = f'{directory}{model}{noise}_results.dictionary'
with open(filepath, 'wb') as parameters:

    pickle.dump(res_BIBFA, parameters)
