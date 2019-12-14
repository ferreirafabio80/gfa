import numpy as np
import numpy.ma as ma
import numpy.linalg as LA
import time
import pickle
import os
from utils import GFAtools
from models.GFA_FA import GFA
from models.GFA_PCA import GFA as GFA_PCA
from scipy.stats import multivariate_normal

#np.random.seed(42)
#Settings
data = 'simulations_lowD'
flag = ''
missing = False
num_init = 5  # number of random initializations
if missing is True:
    p_miss = 40
    remove = 'random'
    scenario = f'missing{str(p_miss)}_{remove}_view2'
    GFAmodel2 = [[] for _ in range(num_init)]
else:
    scenario = 'complete'
model = 'GFA'
noise = 'PCA'
m = 15  
directory = f'results/{data}{flag}/{noise}/{m}models/{scenario}/'
if not os.path.exists(directory):
        os.makedirs(directory)
GFAmodel = [[] for _ in range(num_init)]        
for init in range(0, num_init):
    print("Run:", init+1)

    # Generate some data from the model, with pre-specified
    # latent components
    S = 2  #sources
    Ntrain = 200
    Ntest = 100
    N = Ntrain + Ntest
    d = np.array([25, 15]) # dimensions
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
        #tau[1] = np.array([10,8,7,5,1,1,1])
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
    meanX = np.array((np.mean(X[0]),np.mean(X[1]))) 
    
    # Incomplete data
    #------------------------------------------------------------------------
    if missing is True:
        if 'random' in remove:
            """ for i in range(0,2):
                missing =  np.random.choice([0, 1], size=(X[0].shape[0],d[i]), p=[1-p_miss, p_miss])
                X[i][missing == 1] = 'NaN' """
            missing_val =  np.random.choice([0, 1], size=(X[1].shape[0],d[i]), p=[1-p_miss/100, p_miss/100])
            mask_miss =  ma.array(X[1], mask = missing_val).mask
            missing_true = np.where(missing_val==1,X[1],0)
            X[1][mask_miss] = 'NaN'
        elif 'rows' in remove:
            n_rows = int(p_miss * Ntrain/100)
            nan_ind = np.random.choice(Ntrain, n_rows)
            missing_true = np.zeros((Ntrain,d[1]))
            missing_true[nan_ind,:] = X[1][nan_ind,:]
            X[1][nan_ind,:] = 'NaN' 

    time_start = time.process_time()
    GFAmodel[init] = GFA_PCA(X, m, d)
    L = GFAmodel[init].fit(X)
    GFAmodel[init].L = L
    GFAmodel[init].Z = Z_train
    GFAmodel[init].W = W
    GFAmodel[init].time_elapsed = (time.process_time() - time_start)

    #-Predictions 
    #---------------------------------------------------------------------
    obs_view1 = np.array([0, 1])
    obs_view2 = np.array([1, 0])
    vpred1 = np.array(np.where(obs_view1 == 0))
    vpred2 = np.array(np.where(obs_view2 == 0))
    X_pred = [[] for _ in range(d.size)]
    sig_pred = [[] for _ in range(d.size)]
    X_predmean = [[] for _ in range(d.size)]
    X_pred[vpred1[0,0]], sig_pred[vpred1[0,0]] = GFAtools(X_test, GFAmodel[init], obs_view1).PredictView(noise)
    X_pred[vpred2[0,0]], sig_pred[vpred2[0,0]] = GFAtools(X_test, GFAmodel[init], obs_view2).PredictView(noise)
    X_predmean[vpred1[0,0]] = meanX[vpred1[0,0]] * np.ones((Ntest,d[vpred1[0,0]]))
    X_predmean[vpred2[0,0]] = meanX[vpred2[0,0]] * np.ones((Ntest,d[vpred2[0,0]]))

    #-Metrics
    #----------------------------------------------------------------------------------
    probs = [np.zeros((1,Ntest)) for _ in range(d.size)]
    for j in range(Ntest):
        probs[vpred1[0,0]][0,j] = multivariate_normal.pdf(X_test[vpred1[0,0]][j,:], 
            mean=X_pred[vpred1[0,0]][j,:], cov=sig_pred[vpred1[0,0]])
        probs[vpred2[0,0]][0,j] = multivariate_normal.pdf(X_test[vpred2[0,0]][j,:], 
            mean=X_pred[vpred2[0,0]][j,:], cov=sig_pred[vpred2[0,0]])       
    
    sum_probs = np.sum(probs[0])

    A1 = X_test[vpred1[0,0]] - X_pred[vpred1[0,0]]
    A2 = X_test[vpred1[0,0]] - X_predmean[vpred1[0,0]]
    GFAmodel[init].Fnorm1 = np.sqrt(np.trace(np.dot(A1,A1.T)))
    GFAmodel[init].Fnorm_mean1 = np.sqrt(np.trace(np.dot(A2,A2.T)))

    A1 = X_test[vpred2[0,0]] - X_pred[vpred2[0,0]]
    A2 = X_test[vpred2[0,0]] - X_predmean[vpred2[0,0]]
    GFAmodel[init].Fnorm2 = np.sqrt(np.trace(np.dot(A1,A1.T)))
    GFAmodel[init].Fnorm_mean2 = np.sqrt(np.trace(np.dot(A2,A2.T)))
    
    #relative MSE for each dimension - predict view 1 from view 2
    reMSE = np.zeros((1, d[vpred1[0,0]]))
    reMSEmean = np.zeros((1, d[vpred1[0,0]]))
    for j in range(d[vpred1[0,0]]):
        reMSE[0,j] = np.mean((X_test[vpred1[0,0]][:,j] - X_pred[vpred1[0,0]][:,j]) ** 2)/ np.mean(X_test[vpred1[0,0]][:,j] ** 2)
        reMSEmean[0,j] = np.mean((X_test[vpred1[0,0]][:,j] - X_predmean[vpred1[0,0]][:,j]) ** 2)/ np.mean(X_test[vpred1[0,0]][:,j] ** 2)
    GFAmodel[init].reMSE1 = reMSE
    GFAmodel[init].reMSEmean1 = reMSEmean    

    #relative MSE for each dimension - predict view 2 from view 1
    reMSE = np.zeros((1, d[vpred2[0,0]]))
    reMSEmean = np.zeros((1, d[vpred2[0,0]]))
    for j in range(d[vpred2[0,0]]):
        reMSE[0,j] = np.mean((X_test[vpred2[0,0]][:,j] - X_pred[vpred2[0,0]][:,j]) ** 2)/ np.mean(X_test[vpred2[0,0]][:,j] ** 2)
        reMSEmean[0,j] = np.mean((X_test[vpred2[0,0]][:,j] - X_predmean[vpred2[0,0]][:,j]) ** 2)/ np.mean(X_test[vpred2[0,0]][:,j] ** 2)
    GFAmodel[init].reMSE2 = reMSE
    GFAmodel[init].reMSEmean2 = reMSEmean

    if missing is True:
        miss_view = np.array([1, 0])
        mpred = np.array(np.where(miss_view == 0))
        missing_pred = GFAtools(X, GFAmodel[init],miss_view).PredictMissing()
        #predict missing values
        if 'random' in remove:
            miss_true = missing_true[mask_miss]
            miss_pred = missing_pred[mpred[0,0]][mask_miss]
        elif 'rows' in remove:
            miss_true = missing_true[nan_ind,:]
            miss_pred = missing_pred[mpred[0,0]][nan_ind,:]
        GFAmodel[init].MSEmissing = np.mean((miss_true - miss_pred) ** 2)

        #imputing the predicted missing values in the original matrix,
        #run the model again and make predictions again
        if 'random' in remove:
            X[mpred[0,0]][mask_miss] = miss_pred
        elif 'rows' in remove:
            X[mpred[0,0]][nan_ind,:] = miss_pred    
        GFAmodel2[init] = GFA(X, m, d)
        L = GFAmodel2[init].fit(X)
        GFAmodel2[init].L = L
        X_pred1 = GFAtools(X_test, GFAmodel2[init], obs_view1).PredictView(noise)
        X_pred2 = GFAtools(X_test, GFAmodel2[init], obs_view2).PredictView(noise)

        #-Metrics
        #----------------------------------------------------------------------------------
        #Frobenius norm
        A1 = X_test[vpred1[0,0]] - X_pred1[vpred1[0,0]]    
        GFAmodel2[init].Fnorm1 = np.sqrt(np.trace(np.dot(A1,A1.T)))

        A1 = X_test[vpred2[0,0]] - X_pred2[vpred2[0,0]]
        GFAmodel2[init].Fnorm2 = np.sqrt(np.trace(np.dot(A1,A1.T)))

        #relative MSE for each dimension - view 1
        reMSE = np.zeros((1, d[vpred1[0,0]]))
        for j in range(d[vpred1[0,0]]):
            reMSE[0,j] = np.mean((X_test[vpred1[0,0]][:,j] - X_pred1[vpred1[0,0]][:,j]) ** 2)/ np.mean(X_test[vpred1[0,0]][:,j] ** 2)
        GFAmodel2[init].reMSE1 = reMSE

        #relative MSE for each dimension - view 2 
        reMSE = np.zeros((1, d[vpred2[0,0]]))
        for j in range(d[vpred2[0,0]]):
            reMSE[0,j] = np.mean((X_test[vpred2[0,0]][:,j] - X_pred2[vpred2[0,0]][:,j]) ** 2)/ np.mean(X_test[vpred2[0,0]][:,j] ** 2)
        GFAmodel2[init].reMSE2 = reMSE

        #Save file
        filepath = f'{directory}{model}_results_imputation.dictionary'
        with open(filepath, 'wb') as parameters:

            pickle.dump(GFAmodel2, parameters)    

#Save file
filepath = f'{directory}{model}_results.dictionary'
with open(filepath, 'wb') as parameters:

    pickle.dump(GFAmodel, parameters)


