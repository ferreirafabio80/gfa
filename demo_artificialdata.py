import numpy as np
import math 
from models.GFA_FA import GFA
import time
import matplotlib.pyplot as plt
import pickle
import os

def GFApredict(X, model, view):
    train = np.where(view == 1)
    pred = np.where(view == 0)
    if not pred[0].size:
        pred = np.array(range(0,model.s))
    else:
        pred = pred[0]    
    N = X[0].shape[0] #number of samples

    # Estimate the covariance of the latent variables
    sigmaZ = np.identity(model.m)
    for i in range(train[0].shape[0]): 
        sigmaZ = sigmaZ + model.E_tau[train[0][i]] * model.E_WW[train[0][i]]

    # Estimate the latent variables       
    w, v = np.linalg.eig(sigmaZ)
    sigmaZ = np.dot(v * np.outer(np.ones((1,model.m)), 1/w), v.T)
    meanZ = np.zeros((N,model.m))
    for i in range(train[0].shape[0]): 
        meanZ = meanZ + np.dot(X[train[0][i]], model.means_w[train[0][i]]) * model.E_tau[train[0][i]]
    meanZ = np.dot(meanZ, sigmaZ)

    # Add a tiny amount of noise on top of the latent variables,
    # to supress possible artificial structure in components that
    # have effectively been turned off
    noise = 1e-05
    meanZ = meanZ + noise * \
        np.dot(np.reshape(np.random.normal(
            0, 1, N * model.m),(N, model.m)), np.linalg.cholesky(sigmaZ)) 

    X_pred = [[] for _ in range(model.s)]
    for j in range(pred.shape[0]):
        X_pred[pred[j]] = np.dot(meanZ, model.means_w[pred[j]].T)          

    return X_pred, meanZ

def missingpred(X, model, view):
    train = np.array(np.where(view == 1))
    pred = np.array(np.where(view == 0))   
    N = X[0].shape[0] #number of samples

    # Estimate the covariance of the latent variables
    sigmaZ = np.identity(model.m)
    for i in range(0, train[0].shape[0]): 
        sigmaZ = sigmaZ + np.mean(model.E_tau[train[0,i]]) * model.E_WW[train[0,i]]

    # Estimate the latent variables       
    w, v = np.linalg.eig(sigmaZ)
    sigmaZ = np.dot(v * np.outer(np.ones((1,model.m)), 1/w), v.T)
    meanZ = np.zeros((N,model.m))
    for i in range(0, train.shape[0]): 
        meanZ = meanZ + np.dot(X[train[0,i]], model.means_w[train[0,i]]) * np.mean(model.E_tau[train[0,i]])
    meanZ = np.dot(meanZ, sigmaZ)

    # Add a tiny amount of noise on top of the latent variables,
    # to supress possible artificial structure in components that
    # have effectively been turned off
    noise = 1e-05
    meanZ = meanZ + noise * \
        np.dot(np.reshape(np.random.normal(
            0, 1, N * model.m),(N, model.m)), np.linalg.cholesky(sigmaZ)) 

    X_pred = [[] for _ in range(model.s)]
    for i in range(0, pred.shape[0]):
        X_pred[pred[0,i]] = np.zeros((N, model.d[pred[0,i]]))
        for n in range(0, X[pred[0,i]].shape[0]):
            for j in range(0, X[pred[0,i]].shape[1]):
                if np.isnan(X[pred[0,i]][n,j]):
                    X_pred[pred[0,i]][n,j] = np.dot(meanZ[n,:], model.means_w[pred[0,i]][j,:].T)          

    return X_pred

np.random.seed(42)
#Settings
data = 'simulations_lowD'
flag = ''
scenario = 'missing20_view2'
model = 'GFA'
noise = 'PCA'
m = 10  
directory = f'results/{data}{flag}/{noise}/{m}models/{scenario}/'
if not os.path.exists(directory):
        os.makedirs(directory)

missing = True
num_init = 10  # number of random initializations
GFAmodel = [[] for _ in range(num_init)]
for init in range(0, num_init):
    print("Run:", init+1)

    # Generate some data from the model, with pre-specified
    # latent components
    S = 2  #sources
    Ntrain = Ntest = 200
    N = Ntrain + Ntest
    d = np.array([15, 8]) # dimensions
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
        #tau[0] = np.array([12,11,10,9,1,1,1,1,1,1,1,1,1,1,1])
        #tau[1] = np.array([7,6,5,4,1,1,1])
        tau[0] = 6 * np.ones((1,d[0]))[0]
        tau[1] = 3 * np.ones((1,d[1]))[0]
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
        p_miss = 0.20
        """ for i in range(0,2):
            missing =  np.random.choice([0, 1], size=(X[0].shape[0],d[i]), p=[1-p_miss, p_miss])
            X[i][missing == 1] = 'NaN' """
        missing =  np.random.choice([0, 1], size=(X[0].shape[0],d[i]), p=[1-p_miss, p_miss])
        X[1][missing == 1] = 'NaN'   

    time_start = time.process_time()
    GFAmodel[init] = GFA(X, m, d)
    L = GFAmodel[init].fit(X)
    GFAmodel[init].L = L
    GFAmodel[init].Z = Z_train
    GFAmodel[init].W = W
    GFAmodel[init].time_elapsed = (time.process_time() - time_start)

    if missing is False:
        #Predict one view from the others
        obs_view = np.array([0, 1])
        vpred = np.array(np.where(obs_view == 0))
        X_pred_te, Z_pred = GFApredict(X_test, GFAmodel[init], obs_view)        
        if vpred[0].size:       
            GFAmodel[init].MSEtest = np.mean((X_test[vpred[0]] - X_pred_te[vpred[0]]) ** 2)
        else:
            mse = np.zeros((1,S))
            for i in range(0, S):
                mse[0,i] = np.mean((X_test[i] - X_pred_te[i]) ** 2)
            GFAmodel[init].MSEtest = mse    
            #GFAmodel[init].MSEtrain = np.mean((X_train[mpred] - X_pred_tr[mpred]) ** 2)

        #relative MSE for each dimension
        #error = np.zeros((1, d[mpred]))
        #for d in range(d[mpred]):
        #    error[0,d] = np.mean((X_test[mpred][:,d] - X_pred[mpred][:,d]) ** 2)/ np.mean(X_test[mpred][:,d] ** 2)
    else:
        obs_view = np.array([1, 0])
        X_pred = missingpred(X, GFAmodel[init],obs_view)
        GFAmodel[init].MSEtest = np.mean((X_test[vpred[0]] - X_pred_te[vpred[0]]) ** 2)
            
#Save file
filepath = f'{directory}{model}_results.dictionary'
with open(filepath, 'wb') as parameters:

    pickle.dump(GFAmodel, parameters)

