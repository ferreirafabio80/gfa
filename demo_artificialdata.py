import numpy as np
import numpy.ma as ma
import numpy.linalg as LA
import time
import pickle
import os
import copy
from utils import GFAtools
from models.GFA import GFA_original, GFA_incomplete
from visualization_paper import results_simulations

#Settings
data = 'simulations_paper'
flag = 'lowD'
noise = 'FA'
k = 15
num_init = 10  # number of random initializations
missing = True
prediction = True
perc_train = 80
if missing:
    p_miss = [20]
    remove = ['random'] 
    vmiss = [2]
    if len(remove) == 2:
        scenario = f'missing_v{str(vmiss[0])}{remove[0]}{str(p_miss[0])}_v{str(vmiss[1])}{remove[1]}{str(p_miss[1])}'
    else:
        scenario = f'missing_v{str(vmiss[0])}{remove[0]}{str(p_miss[0])}'    
    if prediction:
        GFAmodel2 = [[] for _ in range(num_init)]
        GFAmodel3 = [[] for _ in range(num_init)]
else:
    scenario = 'complete'

if prediction:
    split_data = f'training{str(perc_train)}'
else:
    split_data = 'all'

directory = f'results/{data}/{flag}/{noise}/{k}models/{scenario}/{split_data}'
if not os.path.exists(directory):
        os.makedirs(directory)

file_path = f'{directory}/GFA_results.dictionary'
if not os.path.exists(file_path):
    GFAmodel = [[] for _ in range(num_init)]        
    for init in range(0, num_init):
        print("Run:", init+1)

        # Generate some data from the model, with pre-specified
        # latent components
        M = 2  #sources
        Ntrain = 400
        Ntest = 100
        N = Ntrain + Ntest
        d = np.array([50, 30]) # dimensions
        T = 4                 # components
        Z = np.zeros((N, T))
        j = 0
        for i in range(0, N):
            Z[i,0] = np.sin((i+1)/(N/20))
            Z[i,1] = np.cos((i+1)/(N/20))
            Z[i,2] = 2*((i+1)/N-0.5)       
        Z[:,3] = np.random.normal(0, 1, N)

        #Diagonal noise precisions
        tau = [[] for _ in range(d.size)]
        if 'FA' in noise:
            tau[0] = 3 * np.ones((1,d[0]))[0] 
            tau[1] = 6 * np.ones((1,d[1]))[0]
        else:    
            tau[0] = 3 * np.ones((1,d[0]))[0]
            tau[1] = 6 * np.ones((1,d[1]))[0]

        #ARD parameters
        alpha = np.zeros((M, T))
        alpha[0,:] = np.array([1,1,1e8,1])
        alpha[1,:] = np.array([1,1,1,1e8])

        #Sample data
        X = [[] for _ in range(d.size)]
        W = [[] for _ in range(d.size)]
        if prediction:
            X_test = [[] for _ in range(d.size)]
        
        for i in range(0, d.size):
            W[i] = np.zeros((d[i], T))
            for t in range(0, T):
                W[i][:,t] = np.random.normal(0, 1/np.sqrt(alpha[i,t]), d[i])
            
            X[i] = np.zeros((N, d[i]))
            for j in range(0, d[i]):
                X[i][:,j] = np.dot(Z,W[i][j,:].T) + \
                np.random.normal(0, 1/np.sqrt(tau[i][j]), N*1)    
            
            if prediction:
                X_test[i] = X[i][Ntrain:N,:] #Test data
                X[i] = X[i][0:Ntrain,:] #Train data

        if prediction:    
            Z = Z[0:Ntrain,:]
            Z_test = Z[Ntrain:N,:]
        
        # Incomplete data
        #------------------------------------------------------------------------
        if missing:
            mask_miss = [[] for _ in range(len(remove))]
            missing_true = [[] for _ in range(len(remove))]
            samples = [[] for _ in range(len(remove))]
            X_median = [[] for _ in range(len(remove))]
            for i in range(len(remove)):
                if 'random' in remove[i]:
                    missing_val =  np.random.choice([0, 1], 
                                size=(X[vmiss[i]-1].shape[0],d[vmiss[i]-1]), p=[1-p_miss[i-1]/100, p_miss[i-1]/100])
                    mask_miss[i] =  ma.array(X[vmiss[i]-1], mask = missing_val).mask
                    missing_true[i] = np.where(missing_val==1, X[vmiss[i]-1],0)
                    X[vmiss[i]-1][mask_miss[i]] = 'NaN'
                elif 'rows' in remove[i]:
                    n_rows = int(p_miss[i-1]/100 * X[vmiss[i]-1].shape[0])
                    samples[i] = np.arange(X[vmiss[i]-1].shape[0])
                    np.random.shuffle(samples[i])
                    missing_true[i] = np.ndarray.flatten(X[vmiss[i]-1][samples[i][0:n_rows],:])
                    X[vmiss[i]-1][samples[i][0:n_rows],:] = 'NaN'
                elif 'nonrand' in remove[i]:
                    miss_mat = np.zeros((X[vmiss[i]-1].shape[0], X[vmiss[i]-1].shape[1]))
                    miss_mat[X[vmiss[i]-1] > p_miss[i] * np.std(X[vmiss[i]-1])] = 1
                    miss_mat[X[vmiss[i]-1] < - p_miss[i] * np.std(X[vmiss[i]-1])] = 1
                    mask_miss[i] =  ma.array(X[vmiss[i]-1], mask = miss_mat).mask
                    missing_true[i] = np.where(miss_mat==1, X[vmiss[i]-1],0)
                    X[vmiss[i]-1][mask_miss[i]] = 'NaN'
                X_median[i] = np.nanmedian(X[vmiss[i]-1],axis=0)    
            GFAmodel[init] = GFA_incomplete(X, k, d)
        elif 'FA' == noise:   
            GFAmodel[init] = GFA_incomplete(X, k, d)
        else:
            GFAmodel[init] = GFA_original(X, k, d)
        
        time_start = time.process_time()
        L = GFAmodel[init].fit(X)
        GFAmodel[init].L = L
        GFAmodel[init].Z = Z
        GFAmodel[init].W = W
        GFAmodel[init].tau = tau
        GFAmodel[init].d = d
        GFAmodel[init].alphas = alpha
        GFAmodel[init].time_elapsed = (time.process_time() - time_start)
        GFAmodel[init].N_test = Ntest
        GFAmodel[init].k_true = T
        print("Computational time: ", GFAmodel[init].time_elapsed) 

        if missing:
            #predict missing values
            MSE_missing = [[] for _ in range(len(remove))]
            for i in range(len(remove)):
                if vmiss[i] == 1:
                    miss_view = np.array([0, 1])
                elif vmiss[i] == 2:
                    miss_view = np.array([1, 0])
                vpred = np.array(np.where(miss_view == 0))
                missing_pred = GFAtools(X, GFAmodel[init], miss_view).PredictMissing()                
                if 'random' in remove[i] or 'nonrand' in remove[i]:
                    miss_true = missing_true[i][mask_miss[i]]
                    miss_pred = missing_pred[vpred[0,0]][mask_miss[i]]
                elif 'rows' in remove[i]:
                    n_rows = int(p_miss[i-1]/100 * X[vmiss[i]-1].shape[0])
                    miss_true = missing_true[i]
                    miss_pred = missing_pred[vpred[0,0]][samples[i][0:n_rows],:]
                MSE_missing[i] = np.mean((miss_true - np.ndarray.flatten(miss_pred)) ** 2)/np.mean(miss_true ** 2)      
            GFAmodel[init].MSEmissing = MSE_missing 
            GFAmodel[init].remove = remove
            GFAmodel[init].vmiss = vmiss

        if prediction:
            #-Predictions 
            #---------------------------------------------------------------------
            obs_view1 = np.array([0, 1])
            obs_view2 = np.array([1, 0])
            vpred1 = np.array(np.where(obs_view1 == 0))
            vpred2 = np.array(np.where(obs_view2 == 0))
            
            #- MODEL 1 
            #---------------------------------------------------------------------------------- 
            X_pred = [[] for _ in range(d.size)]
            X_pred[vpred1[0,0]] = GFAtools(X_test, GFAmodel[init], obs_view1).PredictView(noise)
            X_pred[vpred2[0,0]] = GFAtools(X_test, GFAmodel[init], obs_view2).PredictView(noise)

            #-Metrics           
            #relative MSE for each dimension - predict view 1 from view 2
            reMSE = np.zeros((1, d[vpred1[0,0]]))
            for j in range(d[vpred1[0,0]]):
                reMSE[0,j] = np.mean((X_test[vpred1[0,0]][:,j] - X_pred[vpred1[0,0]][:,j]) ** 2)/ np.mean(X_test[vpred1[0,0]][:,j] ** 2)
            GFAmodel[init].reMSE1 = reMSE

            #relative MSE for each dimension - predict view 2 from view 1
            reMSE = np.zeros((1, d[vpred2[0,0]]))
            for j in range(d[vpred2[0,0]]):
                reMSE[0,j] = np.mean((X_test[vpred2[0,0]][:,j] - X_pred[vpred2[0,0]][:,j]) ** 2)/ np.mean(X_test[vpred2[0,0]][:,j] ** 2)
            GFAmodel[init].reMSE2 = reMSE

            if missing:
                
                #- MODEL 2 
                #---------------------------------------------------------------------------------- 
                #imputing the predicted missing values in the original matrix,
                #run the model again and make predictions
                #----------------------------------------------------------------------------------
                for i in range(len(remove)):
                    if 'random' in remove[i] or 'nonrand' in remove[i]:
                        X_imp = copy.deepcopy(X) 
                        X_imp[vpred[0,0]][mask_miss[i]] = miss_pred
                    elif 'rows' in remove[i]:
                        X_imp = copy.deepcopy(X) 
                        X_imp[vpred[0,0]][samples[i][0:n_rows],:] = miss_pred        
                
                GFAmodel2[init] = GFA_incomplete(X_imp, k, d)
                L = GFAmodel2[init].fit(X_imp)
                GFAmodel2[init].L = L
                GFAmodel2[init].k_true = T
                X_pred = [[] for _ in range(d.size)]
                X_pred[vpred1[0,0]] = GFAtools(X_test, GFAmodel2[init], obs_view1).PredictView(noise)
                X_pred[vpred2[0,0]] = GFAtools(X_test, GFAmodel2[init], obs_view2).PredictView(noise)

                #-Metrics
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

                #- MODEL 3
                #---------------------------------------------------------------------------------- 
                #impute median, run the model again and make predictions
                #----------------------------------------------------------------------------------
                X_impmed = copy.deepcopy(X) 
                for i in range(len(remove)):
                    for j in range(X_median[i].size):
                        X_impmed[vpred[0,0]][np.isnan(X_impmed[vpred[0,0]][:,j]),j] = X_median[i][j]
                
                GFAmodel3[init] = GFA_incomplete(X_impmed, k, d)
                L = GFAmodel3[init].fit(X_impmed)
                GFAmodel3[init].L = L
                GFAmodel3[init].k_true = T
                X_pred = [[] for _ in range(d.size)]
                X_pred[vpred1[0,0]] = GFAtools(X_test, GFAmodel3[init], obs_view1).PredictView(noise)
                X_pred[vpred2[0,0]] = GFAtools(X_test, GFAmodel3[init], obs_view2).PredictView(noise)

                #-Metrics
                #relative MSE for each dimension - predict view 1 from view 2
                reMSE = np.zeros((1, d[vpred1[0,0]]))
                for j in range(d[vpred1[0,0]]):
                    reMSE[0,j] = np.mean((X_test[vpred1[0,0]][:,j] - X_pred[vpred1[0,0]][:,j]) ** 2)/ np.mean(X_test[vpred1[0,0]][:,j] ** 2)
                GFAmodel3[init].reMSE1 = reMSE
                #relative MSE for each dimension - predict view 2 from view 1 
                reMSE = np.zeros((1, d[vpred2[0,0]]))
                for j in range(d[vpred2[0,0]]):
                    reMSE[0,j] = np.mean((X_test[vpred2[0,0]][:,j] - X_pred[vpred2[0,0]][:,j]) ** 2)/ np.mean(X_test[vpred2[0,0]][:,j] ** 2)
                GFAmodel3[init].reMSE2 = reMSE 

                #Save file
                missing_path = f'{directory}/GFA_results_imputation.dictionary'
                with open(missing_path, 'wb') as parameters:
                    pickle.dump(GFAmodel2, parameters)
                
                #Save file
                median_path = f'{directory}/GFA_results_median.dictionary'
                with open(median_path, 'wb') as parameters:
                    pickle.dump(GFAmodel3, parameters)  

    #Save file
    with open(file_path, 'wb') as parameters:
        pickle.dump(GFAmodel, parameters)

#visualization
results_simulations(directory)



