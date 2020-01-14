import numpy as np
import numpy.ma as ma
import numpy.linalg as LA
import time
import pickle
import os
from utils import GFAtools
from models.GFA_FA import GFA as GFAmissing
from models.GFA_PCA import GFA as GFAcomplete
from scipy.stats import multivariate_normal
from visualization_paper import results_simulations

#Settings
data = 'simulations_paper'
flag = 'lowD'
noise = 'FA'
m = 15
num_init = 5  # number of random initializations
missing = True
prediction = True
if missing:
    p_miss = 20
    remove = ['random'] #'random'
    vmiss = [2] #2
    if len(remove) == 2:
        scenario = f'missing{str(p_miss)}_{remove[0]}{remove[1]}_both'
    else:
        scenario = f'missing{str(p_miss)}_{remove[0]}_view{str(vmiss[0])}'    
    if prediction:
        GFAmodel2 = [[] for _ in range(num_init)]
else:
    scenario = 'complete'

if prediction:
    perc_train = 80 
    split_data = f'training{perc_train}'
else:
    split_data = 'all'

directory = f'results/{data}/{flag}/{noise}/{m}models/{scenario}/{split_data}'
if not os.path.exists(directory):
        os.makedirs(directory)

file_path = f'{directory}/GFA_results.dictionary'
if not os.path.exists(file_path):
    GFAmodel = [[] for _ in range(num_init)]        
    for init in range(0, num_init):
        print("Run:", init+1)

        # Generate some data from the model, with pre-specified
        # latent components
        S = 2  #sources
        Ntrain = 200
        Ntest = 100
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
        if 'FA' in noise:
            tau[0] = np.array([12,11,10,9,5,3,2,1,1,1,1,1,1,1,1])
            tau[1] = np.array([10,8,7,5,2,1,1,1])
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
        if missing:
            for i in range(len(remove)):
                if 'random' in remove[i]:
                    missing_val =  np.random.choice([0, 1], size=(X[vmiss[i]-1].shape[0],d[vmiss[i]-1]), p=[1-p_miss/100, p_miss/100])
                    mask_miss =  ma.array(X[vmiss[i]-1], mask = missing_val).mask
                    missing_true = np.where(missing_val==1, X[vmiss[i]-1],0)
                    X[vmiss[i]-1][mask_miss] = 'NaN'
                elif 'rows' in remove[i]:
                    n_rows = int(p_miss/100 * X[vmiss[i]-1].shape[0])
                    samples = np.arange(X[vmiss[i]-1].shape[0])
                    np.random.shuffle(samples)
                    missing_true = np.zeros((Ntrain,d[vmiss[i]-1]))
                    missing_true[samples[0:n_rows],:] = X[vmiss[i]-1][samples[0:n_rows],:]
                    X[vmiss[i]-1][samples[0:n_rows],:] = 'NaN'    
            GFAmodel[init] = GFAmissing(X, m, d)
        elif 'FA' is noise:   
            GFAmodel[init] = GFAmissing(X, m, d)
        else:
            GFAmodel[init] = GFAcomplete(X, m, d)
        
        L = GFAmodel[init].fit(X)
        GFAmodel[init].L = L
        GFAmodel[init].Z = Z_train
        GFAmodel[init].W = W
        GFAmodel[init].alphas = alpha

        if prediction:
            #-Predictions 
            #---------------------------------------------------------------------
            obs_view1 = np.array([0, 1])
            obs_view2 = np.array([1, 0])
            vpred1 = np.array(np.where(obs_view1 == 0))
            vpred2 = np.array(np.where(obs_view2 == 0))
            X_pred = [[] for _ in range(d.size)]
            sig_pred = [[] for _ in range(d.size)]
            X_predmean = [[] for _ in range(d.size)]
            X_pred[vpred1[0,0]] = GFAtools(X_test, GFAmodel[init], obs_view1).PredictView(noise)
            X_pred[vpred2[0,0]] = GFAtools(X_test, GFAmodel[init], obs_view2).PredictView(noise)
            X_predmean[vpred1[0,0]] = meanX[vpred1[0,0]] * np.ones((Ntest,d[vpred1[0,0]]))
            X_predmean[vpred2[0,0]] = meanX[vpred2[0,0]] * np.ones((Ntest,d[vpred2[0,0]]))

            #-Metrics
            #----------------------------------------------------------------------------------
            """ probs = [np.zeros((1,Ntest)) for _ in range(d.size)]
            for j in range(Ntest):
                probs[vpred1[0,0]][0,j] = multivariate_normal.pdf(X_test[vpred1[0,0]][j,:], 
                    mean=X_pred[vpred1[0,0]][j,:], cov=sig_pred[vpred1[0,0]])
                probs[vpred2[0,0]][0,j] = multivariate_normal.pdf(X_test[vpred2[0,0]][j,:], 
                    mean=X_pred[vpred2[0,0]][j,:], cov=sig_pred[vpred2[0,0]])       
            
            sum_probs = np.sum(probs[0]) """
            
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

            if missing:
                if vmiss[0] == 1:
                    miss_view = np.array([0, 1])
                elif vmiss[0] == 2:
                    miss_view = np.array([1, 0])
                vpred = np.array(np.where(miss_view == 0))
                missing_pred = GFAtools(X, GFAmodel[init], miss_view).PredictMissing()
                #predict missing values
                if 'random' in remove:
                    miss_true = missing_true[mask_miss]
                    miss_pred = missing_pred[vpred[0,0]][mask_miss]
                elif 'rows' in remove:
                    miss_true = missing_true[samples,:]
                    miss_pred = missing_pred[vpred[0,0]][samples,:]
                GFAmodel[init].MSEmissing = np.mean((miss_true - miss_pred) ** 2)

                #imputing the predicted missing values in the original matrix,
                #run the model again and make predictions
                if 'random' in remove:
                    X[vpred[0,0]][mask_miss] = miss_pred
                elif 'rows' in remove:
                    X[vpred[0,0]][samples,:] = miss_pred    
                GFAmodel2[init] = GFAmissing(X, m, d)
                L = GFAmodel2[init].fit(X)
                GFAmodel2[init].L = L
                X_pred = [[] for _ in range(d.size)]
                X_pred[vpred1[0,0]] = GFAtools(X_test, GFAmodel2[init], obs_view1).PredictView(noise)
                X_pred[vpred2[0,0]] = GFAtools(X_test, GFAmodel2[init], obs_view2).PredictView(noise)

                #-Metrics
                #----------------------------------------------------------------------------------

                #relative MSE for each dimension - predict view 1 from view 2
                reMSE = np.zeros((1, d[vpred1[0,0]]))
                for j in range(d[vpred1[0,0]]):
                    reMSE[0,j] = np.mean((X_test[vpred1[0,0]][:,j] - X_pred[vpred1[0,0]][:,j]) ** 2)/ np.mean(X_test[vpred1[0,0]][:,j] ** 2)
                GFAmodel2[init].reMSE1 = reMSE

                #relative MSE for each dimension - predict view 2 from view 1 
                reMSE = np.zeros((1, d[vpred2[0,0]]))
                for j in range(d[vpred2[0,0]]):
                    reMSE[0,j] = np.mean((X_test[vpred2[0,0]][:,j] - X_pred[vpred2[0,0]][:,j]) ** 2)/ np.mean(X_test[vpred2[0,0]][:,j] ** 2)
                GFAmodel2[init].reMSE2 = reMSE

                #Save file
                missing_path = f'{directory}/GFA_results_imputation.dictionary'
                with open(missing_path, 'wb') as parameters:

                    pickle.dump(GFAmodel2, parameters)    

    #Save file
    with open(file_path, 'wb') as parameters:

        pickle.dump(GFAmodel, parameters)

#visualization
results_simulations(directory)



