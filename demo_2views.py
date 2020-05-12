import numpy as np
import numpy.ma as ma
import numpy.linalg as LA
import time
import pickle
import os
import copy
from utils import GFAtools
from models import GFA
from visualization import results_simulations

#Settings
#create a dictionary with parameters
data = 'simulations_paper'
flag = 'lowD'
noise = 'diagonal' #spherical diagonal
missing = False
k = 10
num_init = 10  # number of random initializations
perc_train = 80
if missing:
    p_miss = [20,20]
    remove = ['rows','random'] 
    vmiss = [1,2]
    if len(remove) == 2:
        scenario = f'missing_v{str(vmiss[0])}{remove[0]}{str(p_miss[0])}_v{str(vmiss[1])}{remove[1]}{str(p_miss[1])}'
        miss_trainval = True
    else:
        scenario = f'missing_v{str(vmiss[0])}{remove[0]}{str(p_miss[0])}' 
        miss_trainval = False   
    GFAmodel2 = [[] for _ in range(num_init)]
    GFAmodel3 = [[] for _ in range(num_init)]
else:
    scenario = 'complete'

split_data = f'training{str(perc_train)}'
res_dir = f'results/{data}/{flag}/GFA_{noise}/{k}models/{scenario}/{split_data}'
if not os.path.exists(res_dir):
        os.makedirs(res_dir)

file_path = f'{res_dir}/Results_{num_init}runs.dictionary'
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
        if flag == 'highD':
            d = np.array([20000, 200])
        else:
            d = np.array([50, 30])
        T = 4               # components
        Z = np.zeros((N, T))
        j = 0
        for i in range(0, N):
            Z[i,0] = np.sin((i+1)/(N/20))
            Z[i,1] = np.cos((i+1)/(N/20))
            Z[i,2] = 2 * ((i+1)/N-0.5)
        for j in range(3,T):    
            Z[:,j] = np.random.normal(0, 1, N)          

        #Diagonal noise precisions
        tau = [[] for _ in range(d.size)]
        tau[0] = 5 * np.ones((1,d[0]))[0] 
        tau[1] = 10 * np.ones((1,d[1]))[0]

        #ARD parameters
        alpha = np.zeros((M, T))
        alpha[0,:] = np.array([1,1,1e6,1])
        alpha[1,:] = np.array([1,1,1,1e6])
        #alpha[0,:] = np.array([1,1,1e6,1,1e3,1e3,1e6,1e3])
        #alpha[1,:] = np.array([1,1,1,1e6,1e6,1e6,1e3,1e3])      

        #Sample data
        W = [[] for _ in range(d.size)]
        X_train = [[] for _ in range(d.size)]
        X_test = [[] for _ in range(d.size)]
        
        for i in range(0, d.size):
            W[i] = np.zeros((d[i], T))
            for t in range(0, T):
                W[i][:,t] = np.random.normal(0, 1/np.sqrt(alpha[i,t]), d[i])
            
            X = np.zeros((N, d[i]))
            for j in range(0, d[i]):
                X[:,j] = np.dot(Z,W[i][j,:].T) + \
                np.random.normal(0, 1/np.sqrt(tau[i][j]), N*1)    
            
            X_train[i] = X[0:Ntrain,:] #Train data
            X_test[i] = X[Ntrain:N,:] #Test data

        #Latent variables for training the model    
        Z = Z[0:Ntrain,:]
        
        # Incomplete data
        #------------------------------------------------------------------------
        if 'diagonal' in noise: 
            if missing:
                mask_miss = [[] for _ in range(len(remove))]
                missing_true = [[] for _ in range(len(remove))]
                samples = [[] for _ in range(len(remove))]
                X_median = [[] for _ in range(len(remove))]
                for i in range(len(remove)):
                    if 'random' in remove[i]:
                        missing_val =  np.random.choice([0, 1], 
                                    size=(X_train[vmiss[i]-1].shape[0],d[vmiss[i]-1]), p=[1-p_miss[i-1]/100, p_miss[i-1]/100])
                        mask_miss[i] =  ma.array(X_train[vmiss[i]-1], mask = missing_val).mask
                        missing_true[i] = np.where(missing_val==1, X_train[vmiss[i]-1],0)
                        X_train[vmiss[i]-1][mask_miss[i]] = 'NaN'
                    elif 'rows' in remove[i]:
                        n_rows = int(p_miss[i-1]/100 * X_train[vmiss[i]-1].shape[0])
                        samples[i] = np.arange(X_train[vmiss[i]-1].shape[0])
                        np.random.shuffle(samples[i])
                        missing_true[i] = np.ndarray.flatten(X_train[vmiss[i]-1][samples[i][0:n_rows],:])
                        X_train[vmiss[i]-1][samples[i][0:n_rows],:] = 'NaN'
                    elif 'nonrand' in remove[i]:
                        miss_mat = np.zeros((X_train[vmiss[i]-1].shape[0], X_train[vmiss[i]-1].shape[1]))
                        miss_mat[X_train[vmiss[i]-1] > p_miss[i] * np.std(X_train[vmiss[i]-1])] = 1
                        miss_mat[X_train[vmiss[i]-1] < - p_miss[i] * np.std(X_train[vmiss[i]-1])] = 1
                        mask_miss[i] =  ma.array(X_train[vmiss[i]-1], mask = miss_mat).mask
                        missing_true[i] = np.where(miss_mat==1, X_train[vmiss[i]-1],0)
                        X_train[vmiss[i]-1][mask_miss[i]] = 'NaN'                   
                    X_median[i] = np.nanmedian(X_train[vmiss[i]-1],axis=0)    
            GFAmodel[init] = GFA.MissingModel(X_train, k)
        else:
            assert missing is False
            GFAmodel[init] = GFA.OriginalModel(X_train, k)
        
        time_start = time.process_time()
        L = GFAmodel[init].fit(X_train)
        GFAmodel[init].time_elapsed = (time.process_time() - time_start)
        GFAmodel[init].L = L
        GFAmodel[init].Z = Z
        GFAmodel[init].W = W
        GFAmodel[init].tau = tau
        GFAmodel[init].d = d
        GFAmodel[init].alphas = alpha
        GFAmodel[init].N_test = Ntest
        GFAmodel[init].k_true = T
        GFAmodel[init].train_perc = perc_train
        GFAmodel[init].X = X_train
        print("Computational time: ", GFAmodel[init].time_elapsed)    
        
        if missing:
            #predict missing values
            MSE_missing = [[] for _ in range(len(remove))]
            miss_pred = [[] for _ in range(len(remove))]
            for i in range(len(remove)):
                if vmiss[i] == 1:
                    miss_view = np.array([0, 1])
                elif vmiss[i] == 2:
                    miss_view = np.array([1, 0])
                vpred = np.array(np.where(miss_view == 0))                
                if 'random' in remove[i] or 'nonrand' in remove[i]:
                    missing_pred = GFAtools(X_train, GFAmodel[init], miss_view).PredictMissing(missTrain=miss_trainval)
                    miss_true = missing_true[i][mask_miss[i]]
                    miss_pred[i] = missing_pred[vpred[0,0]][mask_miss[i]]
                elif 'rows' in remove[i]:
                    missing_pred = GFAtools(X_train, GFAmodel[init], miss_view).PredictMissing(missTrain=miss_trainval,missRows=True)
                    n_rows = int(p_miss[i-1]/100 * X_train[vmiss[i]-1].shape[0])
                    miss_true = missing_true[i]
                    miss_pred[i] = missing_pred[vpred[0,0]][samples[i][0:n_rows],:]
                MSE_missing[i] = np.mean((miss_true - np.ndarray.flatten(miss_pred[i])) ** 2)   
            GFAmodel[init].MSEmissing = MSE_missing 
            GFAmodel[init].remove = remove
            GFAmodel[init].vmiss = vmiss

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
        #MSE - predict view 1 from view 2
        MSE1 = np.mean((X_test[vpred1[0,0]] - X_pred[vpred1[0,0]]) ** 2)
        GFAmodel[init].MSE1 = MSE1
        #MSE - predict view 2 from view 1
        MSE2 = np.mean((X_test[vpred2[0,0]] - X_pred[vpred2[0,0]]) ** 2)
        GFAmodel[init].MSE2 = MSE2

        if missing:
            
            #- MODEL 2 
            #---------------------------------------------------------------------------------- 
            #imputing the predicted missing values in the original matrix,
            #run the model again and make predictions
            #----------------------------------------------------------------------------------
            print("Imputation Model----------") 
            X_imp = copy.deepcopy(X_train) 
            for i in range(len(remove)):
                if vmiss[i] == 1:
                    miss_view = np.array([0, 1])
                elif vmiss[i] == 2:
                    miss_view = np.array([1, 0])
                vpred = np.array(np.where(miss_view == 0))
                if 'random' in remove[i] or 'nonrand' in remove[i]:
                    X_imp[vpred[0,0]][mask_miss[i]] = np.ndarray.flatten(miss_pred[i])
                elif 'rows' in remove[i]:
                    X_imp[vpred[0,0]][samples[i][0:n_rows],:] = miss_pred[i]        
            
            GFAmodel2[init] = GFA.OriginalModel(X_imp, k)
            L = GFAmodel2[init].fit(X_imp)
            GFAmodel2[init].L = L
            GFAmodel2[init].k_true = T
            X_pred = [[] for _ in range(d.size)]
            X_pred[vpred1[0,0]] = GFAtools(X_test, GFAmodel2[init], obs_view1).PredictView('spherical')
            X_pred[vpred2[0,0]] = GFAtools(X_test, GFAmodel2[init], obs_view2).PredictView('spherical')

            #-Metrics
            #MSE - predict view 1 from view 2
            MSE1 = np.mean((X_test[vpred1[0,0]] - X_pred[vpred1[0,0]]) ** 2)    
            GFAmodel2[init].MSE1 = MSE1
            #MSE - predict view 2 from view 1 
            MSE2 = np.mean((X_test[vpred2[0,0]] - X_pred[vpred2[0,0]]) ** 2)    
            GFAmodel2[init].MSE2 = MSE2

            #- MODEL 3
            #---------------------------------------------------------------------------------- 
            #impute median, run the model again and make predictions
            #----------------------------------------------------------------------------------
            print("Median Model----------") 
            X_impmed = copy.deepcopy(X_train) 
            for i in range(len(remove)):
                if vmiss[i] == 1:
                    miss_view = np.array([0, 1])
                elif vmiss[i] == 2:
                    miss_view = np.array([1, 0])
                vpred = np.array(np.where(miss_view == 0))
                for j in range(X_median[i].size):
                    X_impmed[vpred[0,0]][np.isnan(X_impmed[vpred[0,0]][:,j]),j] = X_median[i][j]
            
            GFAmodel3[init] = GFA.OriginalModel(X_impmed, k)
            L = GFAmodel3[init].fit(X_impmed)
            GFAmodel3[init].L = L
            GFAmodel3[init].k_true = T
            X_pred = [[] for _ in range(d.size)]
            X_pred[vpred1[0,0]] = GFAtools(X_test, GFAmodel3[init], obs_view1).PredictView('spherical')
            X_pred[vpred2[0,0]] = GFAtools(X_test, GFAmodel3[init], obs_view2).PredictView('spherical')

            #-Metrics
            #MSE - predict view 1 from view 2
            MSE1 = np.mean((X_test[vpred1[0,0]] - X_pred[vpred1[0,0]]) ** 2)    
            GFAmodel3[init].MSE1 = MSE1
            #MSE - predict view 2 from view 1 
            MSE2 = np.mean((X_test[vpred2[0,0]] - X_pred[vpred2[0,0]]) ** 2)    
            GFAmodel3[init].MSE2 = MSE2
  
    if missing:
        #Save file
        missing_path = f'{res_dir}/Results_imputation.dictionary'
        with open(missing_path, 'wb') as parameters:
            pickle.dump(GFAmodel2, parameters)        
        #Save file
        median_path = f'{res_dir}/Results_median.dictionary'
        with open(median_path, 'wb') as parameters:
            pickle.dump(GFAmodel3, parameters)

    #Save file
    with open(file_path, 'wb') as parameters:
        pickle.dump(GFAmodel, parameters)        

#visualization
best_model = results_simulations(num_init, res_dir)

#Run reduced model
ofile = open(f'{res_dir}/reduced_model.txt','w')
S=2
rel_comps = np.arange(4)
for i in range(S):
    best_model.means_w[i] = best_model.means_w[i][:,rel_comps]
best_model.means_z = best_model.means_z[:,rel_comps]
if 'spherical' in noise:
    Redmodel = GFA.OriginalModel(best_model.X, rel_comps.size, lowK_model=best_model)
else:     
    Redmodel = GFA.MissingModel(best_model.X, rel_comps.size, lowK_model=best_model)
L = Redmodel.fit(best_model.X)

print(f'Relevant components:', rel_comps, file=ofile)
print(f'Lower bound full model:', best_model.L[-1], file=ofile)
print(f'Lower bound reduced model: ', L[-1], file=ofile)  

#Bayes factor
BF = np.exp(best_model.L[-1]-L[-1]) 
print(f'Bayes factor: ', BF, file=ofile)
ofile.close()

