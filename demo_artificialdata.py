import numpy as np
import numpy.ma as ma
import numpy.linalg as LA
import math 
from models.GFA_FA import GFA
import time
import matplotlib.pyplot as plt
import pickle
import os

def GFApredict(X, model, view, noise):
    train = np.array(np.where(view == 1))
    pred = np.array(np.where(view == 0)) 
    if not pred[0].size:
        pred = np.array(range(0,model.s))
    else:
        pred = pred[0]    
    N = X[0].shape[0] #number of samples

    # Estimate the covariance of the latent variables
    sigmaZ = np.identity(model.m)
    for i in range(train[0].shape[0]): 
        if 'PCA' in noise:
            sigmaZ = sigmaZ + model.E_tau[train[0][i]] * model.E_WW[train[0][i]]
        else:
            for j in range(model.d[train[0,0]]):
                w = np.reshape(model.means_w[train[0,i]][j,:], (1,model.m))
                ww = model.sigma_w[train[0,i]][:,:,j] + np.dot(w.T, w) 
                sigmaZ = sigmaZ + model.E_tau[train[0,i]][0,j] * ww

    # Estimate the latent variables       
    w, v = np.linalg.eig(sigmaZ)
    sigmaZ = np.dot(v * np.outer(np.ones((1,model.m)), 1/w), v.T)
    meanZ = np.zeros((N,model.m))
    for i in range(train[0].shape[0]):
        if 'PCA' in noise: 
            meanZ = meanZ + np.dot(X[train[0][i]], model.means_w[train[0][i]]) * model.E_tau[train[0][i]]
        else: 
            for j in range(model.d[train[0,0]]):
                w = np.reshape(model.means_w[train[0,i]][j,:], (1,model.m)) 
                x = np.reshape(X[train[0,i]][:,j], (N,1)) 
                meanZ = meanZ + np.dot(x, w) * model.E_tau[train[0,i]][0,j]         
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

    return X_pred

def PredictMissing(X, model, view):
    train = np.array(np.where(view == 1))
    pred = np.array(np.where(view == 0))   
    N = X[0].shape[0] #number of samples

    # Estimate the covariance of the latent variables
    sigmaZ = np.identity(model.m)
    for i in range(0, train[0].shape[0]):
        for j in range(model.d[train[0,0]]):
            w = np.reshape(model.means_w[train[0,i]][j,:], (1,model.m))
            ww = model.sigma_w[train[0,i]][:,:,j] + np.dot(w.T, w) 
            sigmaZ = sigmaZ + model.E_tau[train[0,i]][0,j] * ww

    # Estimate the latent variables       
    w, v = np.linalg.eig(sigmaZ)
    sigmaZ = np.dot(v * np.outer(np.ones((1,model.m)), 1/w), v.T)
    meanZ = np.zeros((N,model.m))
    for i in range(0, train.shape[0]):
        for j in range(model.d[train[0,0]]):
            w = np.reshape(model.means_w[train[0,i]][j,:], (1,model.m)) 
            x = np.reshape(X[train[0,i]][:,j], (N,1)) 
            meanZ = meanZ + np.dot(x, w) * model.E_tau[train[0,i]][0,j] 
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

#np.random.seed(42)
#Settings
data = 'simulations_lowD'
flag = ''
scenario = 'complete_view2'
model = 'GFA'
noise = 'FA'
m = 15  
directory = f'results/{data}{flag}/{noise}/{m}models/{scenario}/'
if not os.path.exists(directory):
        os.makedirs(directory)

missing = False
remove = 'random'
num_init = 10  # number of random initializations
GFAmodel = [[] for _ in range(num_init)]
GFAmodel2 = [[] for _ in range(num_init)]
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
        tau[1] = np.array([10,8,7,5,1,1,1])
        #tau[0] = 6 * np.ones((1,d[0]))[0]
        #tau[1] = 3 * np.ones((1,d[1]))[0]
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
        
        X[i] = np.dot(Z, W[i].T) + np.random.multivariate_normal(
            np.zeros((1, d[i]))[0], np.diag(1/np.sqrt(tau[i])), N)  
        
        X_train[i] = X[i][0:Ntrain,:]
        X_test[i] = X[i][Ntrain:N,:]

    Z_train = Z[0:Ntrain,:]
    Z_test = Z[Ntrain:N,:]  
    X = X_train
    meanX = np.array((np.mean(X[0]),np.mean(X[1]))) 
    
    # Incomplete data
    #------------------------------------------------------------------------
    if missing is True:
        if 'random' in remove:
            p_miss = 0.30
            """ for i in range(0,2):
                missing =  np.random.choice([0, 1], size=(X[0].shape[0],d[i]), p=[1-p_miss, p_miss])
                X[i][missing == 1] = 'NaN' """
            missing_val =  np.random.choice([0, 1], size=(X[1].shape[0],d[i]), p=[1-p_miss, p_miss])
            mask_miss =  ma.array(X[1], mask = missing_val).mask
            missing_true = np.where(missing_val==1,X[1],0)
            X[1][mask_miss] = 'NaN'   

    time_start = time.process_time()
    GFAmodel[init] = GFA(X, m, d)
    L = GFAmodel[init].fit(X)
    GFAmodel[init].L = L
    GFAmodel[init].Z = Z_train
    GFAmodel[init].W = W
    GFAmodel[init].time_elapsed = (time.process_time() - time_start)

    if missing is True:
        miss_view = np.array([1, 0])
        vpred = np.array(np.where(miss_view == 0))
        missing_pred = PredictMissing(X, GFAmodel[init],miss_view)
        #predict missing values
        miss_true = missing_true[mask_miss]
        miss_pred = missing_pred[vpred[0,0]][mask_miss]
        GFAmodel[init].MSEmissing = np.mean((miss_true - miss_pred) ** 2)

        #imputing the predicted missing values in the original matrix,
        #run the model again and make predictions again
        X[vpred[0,0]][mask_miss] = miss_pred
        GFAmodel2[init] = GFA(X, m, d)
        L = GFAmodel2[init].fit(X)
        GFAmodel2[init].L = L
        X_pred2, Z_pred2 = GFApredict(X_test, GFAmodel2[init], obs_view, noise)

        #metrics
        A1 = X_test[vpred[0,0]] - X_pred2[vpred[0,0]]
        A2 = X_test[vpred[0,0]] - X_predmean[vpred[0,0]]
        GFAmodel2[init].Fnorm = np.sqrt(np.trace(np.dot(A1,A1.T)))
        GFAmodel2[init].Fnorm_mean = np.sqrt(np.trace(np.dot(A2,A2.T)))

        #relative MSE for each dimension
        reMSE2 = np.zeros((1, d[vpred[0,0]]))
        reMSEmean2 = np.zeros((1, d[vpred[0,0]]))
        for j in range(d[vpred[0,0]]):
            reMSE2[0,j] = np.mean((X_test[vpred[0,0]][:,j] - X_pred2[vpred[0,0]][:,j]) ** 2)/ np.mean(X_test[vpred[0,0]][:,j] ** 2)
            reMSEmean2[0,j] = np.mean((X_test[vpred[0,0]][:,j] - X_predmean[:,j]) ** 2)/ np.mean(X_test[vpred[0,0]][:,j] ** 2)
        GFAmodel2[init].reMSE = reMSE
        GFAmodel2[init].reMSEmean = reMSEmean

        #Save file
        filepath = f'{directory}{model}_results_imputation.dictionary'
        with open(filepath, 'wb') as parameters:

            pickle.dump(GFAmodel2, parameters)    

    #predict view 2 from view 1
    obs_view1 = np.array([1, 0])
    obs_view2 = np.array([0, 1])
    vpred1 = np.array(np.where(obs_view1 == 0))
    vpred2 = np.array(np.where(obs_view2 == 0))
    X_pred1 = GFApredict(X_test, GFAmodel[init], obs_view1, noise)
    X_predmean1 = meanX[vpred1[0,0]] * np.ones((Ntrain,d[vpred1[0,0]]))
    X_pred2 = GFApredict(X_test, GFAmodel[init], obs_view2, noise)
    X_predmean2 = meanX[vpred2[0,0]] * np.ones((Ntrain,d[vpred2[0,0]]))
    
    #metrics
    A1 = X_test[vpred1[0,0]] - X_pred1[vpred1[0,0]]
    A2 = X_test[vpred1[0,0]] - X_predmean1[vpred1[0,0]]
    GFAmodel[init].Fnorm1 = np.sqrt(np.trace(np.dot(A1,A1.T)))
    GFAmodel[init].Fnorm_mean1 = np.sqrt(np.trace(np.dot(A2,A2.T)))

    #relative MSE for each dimension
    reMSE = np.zeros((1, d[vpred1[0,0]]))
    reMSEmean = np.zeros((1, d[vpred1[0,0]]))
    for j in range(d[vpred1[0,0]]):
        reMSE[0,j] = np.mean((X_test[vpred1[0,0]][:,j] - X_pred1[vpred1[0,0]][:,j]) ** 2)/ np.mean(X_test[vpred1[0,0]][:,j] ** 2)
        reMSEmean[0,j] = np.mean((X_test[vpred1[0,0]][:,j] - X_predmean1[:,j]) ** 2)/ np.mean(X_test[vpred1[0,0]][:,j] ** 2)
    GFAmodel[init].reMSE1 = reMSE
    GFAmodel[init].reMSEmean1 = reMSEmean

    #metrics
    A1 = X_test[vpred2[0,0]] - X_pred2[vpred2[0,0]]
    A2 = X_test[vpred2[0,0]] - X_predmean2[vpred2[0,0]]
    GFAmodel[init].Fnorm2 = np.sqrt(np.trace(np.dot(A1,A1.T)))
    GFAmodel[init].Fnorm_mean2 = np.sqrt(np.trace(np.dot(A2,A2.T)))

    #relative MSE for each dimension
    reMSE = np.zeros((1, d[vpred2[0,0]]))
    reMSEmean = np.zeros((1, d[vpred2[0,0]]))
    for j in range(d[vpred2[0,0]]):
        reMSE[0,j] = np.mean((X_test[vpred2[0,0]][:,j] - X_pred2[vpred2[0,0]][:,j]) ** 2)/ np.mean(X_test[vpred2[0,0]][:,j] ** 2)
        reMSEmean[0,j] = np.mean((X_test[vpred2[0,0]][:,j] - X_predmean2[:,j]) ** 2)/ np.mean(X_test[vpred2[0,0]][:,j] ** 2)
    GFAmodel[init].reMSE2 = reMSE
    GFAmodel[init].reMSEmean2 = reMSEmean

#Save file
filepath = f'{directory}{model}_results.dictionary'
with open(filepath, 'wb') as parameters:

    pickle.dump(GFAmodel, parameters)


