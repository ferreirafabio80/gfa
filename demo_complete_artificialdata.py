# Run Bayesian CCA model
import numpy as np
import math
import GFA
import matplotlib.pyplot as plt
import pickle

num_init = 10  # number of random initializations
res_BIBFA = [[] for _ in range(num_init)]
for init in range(0, num_init):
    print("Iterations:", init+1)

    # Generate some data from the model, with pre-specified
    # latent components
    S = 2  # sources
    Ntrain = Ntest = 100
    N = Ntrain + Ntest
    d = np.array([15, 7])  # dimensions
    K = 4                 # components
    Z = np.zeros((N, K))
    j = 0
    for i in range(0, N):
        Z[i, 0] = np.sin((i+1)/(N/20))
        Z[i, 1] = np.cos((i+1)/(N/20))
        if i < Ntrain:
            Z[i, 3] = 2*((i+1)/Ntrain-0.5)
        else:
            Z[i, 3] = 2*((j+1)/Ntest-0.5)
            j += 1
    Z[:, 2] = np.random.normal(0, 1, N)

    # spherical precisions
    tau = np.array([3, 6])

    # ARD parameters
    alpha = np.zeros((S, K))
    alpha[0, :] = np.array([1, 1, 1e6, 1])
    alpha[1, :] = np.array([1, 1, 1, 1e6])

    X = [[] for _ in range(d.size)]
    X_train = [[] for _ in range(d.size)]
    X_test = [[] for _ in range(d.size)]
    W = [[] for _ in range(d.size)]
    for i in range(0, d.size):
        W[i] = np.zeros((d[i], K))
        for k in range(0, K):
            W[i][:, k] = np.random.normal(0, 1/np.sqrt(alpha[i, k]), d[i])

        X[i] = (np.dot(Z, W[i].T) + np.reshape(
            np.random.normal(0, 1/np.sqrt(tau[i]), N*d[i]), (N, d[i])))
        X_train[i] = X[i][0:Ntrain, :]
        X_test[i] = X[i][Ntrain:N, :]

    Z_train = Z[0:Ntrain, :]
    Z_test = Z[Ntrain:N, :]
    X = X_train

    # Complete data
    # ------------------------------------------------------------------------
    m = 8  # number of models
    res_BIBFA[init] = GFA.BIBFA(X, m, d)
    L = res_BIBFA[init].fit(X)
    res_BIBFA[init].L = L
    res_BIBFA[init].Z = Z_train
    res_BIBFA[init].W = W

data = 'simulations'
noise = 'PCA'
scenario = 'complete'
model = 'GFA'
directory = f'results/{data}/{noise}/{m}models/{scenario}/'
filepath = f'{directory}{model}_results.dictionary'
with open(filepath, 'wb') as parameters:

    pickle.dump(res_BIBFA, parameters)
