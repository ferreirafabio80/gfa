import numpy as np
import math 
from models.GFA_FA import GFAmissing
import matplotlib.pyplot as plt
import pickle
import os

#Settings
data = 'simulations_lowD'
flag = ''
scenario = 'missing30_clinical'
model = 'GFA'
noise = 'FA'
m = 15  # number of models
directory = f'results/{data}{flag}/{noise}/{m}models/{scenario}/'
if not os.path.exists(directory):
        os.makedirs(directory)

missing = True
num_init = 10  # number of random initializations
res_BIBFA = [[] for _ in range(num_init)]
for init in range(0, num_init):
    print("Run:", init+1)

    # Generate some data from the model, with pre-specified
    # latent components
    S = 2  #sources
    Ntrain = Ntest = 200
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

    #Diagonal noise precisions
    tau = [[] for _ in range(d.size)]
    if noise == 'FA':
        tau[0] = np.array([12,11,10,9,1,1,1,1,1,1,1,1,1,1,1])
        tau[1] = np.array([7,6,5,4,1,1,1])
    else:    
        tau[0] = 6 * np.ones((1,d[0]))[0]
        tau[1] = 3 * np.ones((1,d[1]))[0]

    #ARD parameters
    alpha = np.zeros((S, K))
    alpha[0,:] = np.array([1,1,1e6,1])
    alpha[1,:] = np.array([1,1,1,1e6])

    #Sample data
    X = [[] for _ in range(d.size)]
    X_train = [[] for _ in range(d.size)]
    X_test = [[] for _ in range(d.size)]
    W = [[] for _ in range(d.size)]
    for i in range(0, d.size):
        W[i] = np.zeros((d[i], K))
        for k in range(0, K):
            W[i][:,k] = np.random.normal(0, 1/np.sqrt(alpha[i,k]), d[i])
        
        X[i] = np.zeros((N, d[i]))
        for j in range(0, d[i]):
            X[i][:,j] = np.dot(Z,W[i][j,:].T) + \
            np.random.normal(0, 1/np.sqrt(tau[i][j]), N*1)   
        
        X_train[i] = X[i][0:Ntrain,:]
        X_test[i] = X[i][Ntrain:N,:]

    Z_train = Z[0:Ntrain,:]
    Z_test = Z[Ntrain:N,:]  
    X = X_train

    # Incomplete data
    #------------------------------------------------------------------------
    if missing is True:
        p_miss = 0.30
        #for i in range(0,2):
        missing =  np.random.choice([0, 1], size=(X[0].shape[0],d[i]), p=[1-p_miss, p_miss])
        X[1][missing == 1] = 'NaN'

    res_BIBFA[init] = GFAmissing(X, m, d)
    L = res_BIBFA[init].fit(X)
    res_BIBFA[init].L = L
    res_BIBFA[init].Z = Z_train
    res_BIBFA[init].W = W

#Save file
filepath = f'{directory}{model}_results.dictionary'
with open(filepath, 'wb') as parameters:

    pickle.dump(res_BIBFA, parameters)