import numpy as np
import numpy.ma as ma
import time
import pickle
import os
import copy
import GFA 
import argparse
from utils import GFAtools
from visualization import results_simulations

def get_data(args, infoMiss=False):
    # Generate some data from the generative model, with pre-specified
    # latent components
    Ntrain = 400; Ntest = 100
    N = Ntrain + Ntest #number of samples
    M = args.num_sources  #number of groups/data sources
    d = np.array([50, 30]) #number of dimensios for each group
    true_K = 4  # true latent components
    #Manually specify Z 
    Z = np.zeros((N, true_K))
    for i in range(0, N):
        Z[i,0] = np.sin((i+1)/(N/20))
        Z[i,1] = np.cos((i+1)/(N/20))
        Z[i,2] = 2 * ((i+1)/N-0.5)    
    Z[:,3] = np.random.normal(0, 1, N)          
    #Noise precisions
    tau = [[] for _ in range(d.size)]
    tau[0] = 5 * np.ones((1,d[0]))[0] 
    tau[1] = 10 * np.ones((1,d[1]))[0]
    #ARD parameters
    alpha = np.zeros((M, true_K))
    alpha[0,:] = np.array([1,1,1e6,1])
    alpha[1,:] = np.array([1,1,1,1e6])     
    #W and X
    W = [[] for _ in range(d.size)]
    X_train = [[] for _ in range(d.size)]
    X_test = [[] for _ in range(d.size)]
    for i in range(0, d.size):
        W[i] = np.zeros((d[i], true_K))
        for t in range(0, true_K):
            #generate W from p(W|alpha)
            W[i][:,t] = np.random.normal(0, 1/np.sqrt(alpha[i,t]), d[i])
        X = np.zeros((N, d[i]))
        for j in range(0, d[i]):
            #generate X from the generative model
            X[:,j] = np.dot(Z,W[i][j,:].T) + \
            np.random.normal(0, 1/np.sqrt(tau[i][j]), N*1)    
        #Get train and test data
        X_train[i] = X[0:Ntrain,:] #Train data
        X_test[i] = X[Ntrain:N,:] #Test data
    #Latent variables for training the model    
    Z = Z[0:Ntrain,:]

    #Generate incomplete training data
    if args.scenario == 'incomplete':
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
            X_median[i] = np.nanmedian(X_train[vmiss[i]-1],axis=0)
    data = {'X_tr': X_train, 'X_te': X_test, 'W': W, 'Z': Z, 'tau': tau, 'alpha': alpha, 'true_K': true_K}         
    return data        

def main(args):
    #info to generate incomplete data sets
    if args.scenario == 'incomplete':
        infmiss = {'perc': [10, 20], #percentage of missing data 
                'type': ['rows', 'random'], #type of missing data 
                'group': [1, 2]} #groups that will have missing values            
        if len(infmiss['group']) > 1:
            miss_trainval = True

    #Make directory to save the results of the experiments          
    res_dir = f'results/2groups/GFA_{args.noise}/{args.K}components/{args.scenario}'
    if not os.path.exists(res_dir):
            os.makedirs(res_dir)
    #Path to the files where the results and data will be saved
    res_file = f'{res_dir}/Results_{args.num_runs}runs.dictionary'
    data_file = f'{res_dir}/Data_{args.num_runs}runs.dictionary'
    if not os.path.exists(res_file):
        GFAmodel = [[] for _ in range(args.num_runs)]
        simData = [[] for _ in range(args.num_runs)]
        if args.impMedian:
            GFAmodel2 = [[] for _ in range(args.num_runs)]
        for init in range(0, args.num_runs):
            print("Run:", init+1)
            print("Generating data---------")
            if args.scenario == 'complete':
                simData[init] = get_data(args)
            else:
                simData[init] = get_data(args, infmiss) 
            
            #Run model 
            X_tr = simData[init]['X_tr']
            if 'diagonal' in args.noise:    
                GFAmodel[init] = GFA.IncompleteDataModel(X_tr, args.K)
            else:
                assert args.scenario == 'complete'
                GFAmodel[init] = GFA.OriginalModel(X_tr, args.K)
            print("Running the model---------")
            time_start = time.process_time()
            GFAmodel[init].fit(X_tr)
            GFAmodel[init].time_elapsed = time.process_time() - time_start
            print(f'Computational time: {float("{:.2f}".format(GFAmodel[init].time_elapsed))}s' )    
            
            if args.scenario == 'incomplete':
                #predict missing values
                MSE_missing = [[] for _ in range(len(remove))]
                Corr_missing = [[] for _ in range(len(remove))]
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
                    Corr_missing[i] = np.corrcoef(miss_true,np.ndarray.flatten(miss_pred[i]))[0,1]   
                GFAmodel[init].Corrmissing = Corr_missing 
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
            X_pred[vpred2[0,0]] = GFAtools(X_test, GFAmodel[init], obs_view2).PredictView(noise)
                
            #MSE - predict view 2 from view 1
            MSE2 = np.mean((X_test[vpred2[0,0]] - X_pred[vpred2[0,0]]) ** 2)
            Xtr_mean = np.zeros((Ntest,d[vpred2[0,0]]))
            Colmean_tr = np.nanmean(X_train[vpred2[0,0]], axis=0)
            for j in range(d[vpred2[0,0]]):
                Xtr_mean[:,j] = Colmean_tr[j]
            MSE2_train = np.mean((X_test[vpred2[0,0]] - Xtr_mean[vpred2[0,0]]) ** 2)
            GFAmodel[init].MSE2 = MSE2
            GFAmodel[init].MSE2_train = MSE2_train

            if impMedian:
                #- MODEL 2
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
                
                GFAmodel2[init] = GFA.MissingModel(X_impmed, k)
                L = GFAmodel2[init].fit(X_impmed)
                GFAmodel2[init].L = L
                GFAmodel2[init].k_true = T
                X_pred = [[] for _ in range(d.size)]
                X_pred[vpred2[0,0]] = GFAtools(X_test, GFAmodel2[init], obs_view2).PredictView('diagonal')

                #-Metrics
                #MSE - predict view 2 from view 1 
                MSE2 = np.mean((X_test[vpred2[0,0]] - X_pred[vpred2[0,0]]) ** 2)    
                GFAmodel2[init].MSE2 = MSE2
    
        if args.impMedian:    
            #Save file
            median_path = f'{res_dir}/Results_median.dictionary'
            with open(median_path, 'wb') as parameters:
                pickle.dump(GFAmodel2, parameters)

        #Save file with results
        with open(res_file, 'wb') as parameters:
            pickle.dump(GFAmodel, parameters)
        #Save file with generated data
        with open(data_file, 'wb') as parameters:
                pickle.dump(simData, parameters)            

    #visualization
    results_simulations(args.num_runs, res_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GFA with two groups")
    parser.add_argument("--scenario", nargs='?', default='complete', type=str)
    parser.add_argument("--noise", nargs='?', default='spherical', type=str)
    parser.add_argument("--num-sources", nargs='?', default=2, type=int)
    parser.add_argument("--K", nargs='?', default=15, type=int)
    parser.add_argument("--num-runs", nargs='?', default=10, type=int)
    parser.add_argument("--miss-train", nargs='?', default=False, type=bool)
    parser.add_argument("--impMedian", nargs='?', default=False, type=bool)
    args = parser.parse_args()

    main(args)    


