import numpy as np
import numpy.ma as ma
import time
import pickle
import os
import copy
import GFA 
import argparse
import visualization_simulations as vis_simData
from utils import GFAtools

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
        missing_Xtrue = [[] for _ in range(len(infoMiss['group']))]
        for i in range(len(infoMiss['group'])): 
            if 'random' in infoMiss['type'][i]: 
                #remove entries randomly
                missing_val =  np.random.choice([0, 1], 
                            size=(X_train[infoMiss['group'][i]-1].shape[0],d[infoMiss['group'][i]-1]), 
                            p=[1-infoMiss['perc'][i-1]/100, infoMiss['perc'][i-1]/100])
                mask_miss =  ma.array(X_train[infoMiss['group'][i]-1], mask = missing_val).mask
                missing_Xtrue[i] = np.where(missing_val==1, X_train[infoMiss['group'][i]-1],0)
                X_train[infoMiss['group'][i]-1][mask_miss] = 'NaN'
            elif 'rows' in infoMiss['type'][i]: 
                #remove rows randomly
                missing_Xtrue[i] = np.zeros((Ntrain,d[i]))
                n_rows = int(infoMiss['perc'][i-1]/100 * X_train[infoMiss['group'][i]-1].shape[0])
                shuf_samples = np.arange(Ntrain)
                np.random.shuffle(shuf_samples)
                missing_Xtrue[i][shuf_samples[0:n_rows],:] = X_train[infoMiss['group'][i]-1][shuf_samples[0:n_rows],:]
                X_train[infoMiss['group'][i]-1][shuf_samples[0:n_rows],:] = 'NaN'
    #Store data            
    data = {'X_tr': X_train, 'X_te': X_test, 'W': W, 'Z': Z, 'tau': tau, 'alpha': alpha, 'true_K': true_K}
    if args.scenario == 'incomplete':
        data.update({'trueX_miss': missing_Xtrue}) 
    return data        

def main(args):
    #info to generate incomplete data sets
    if args.scenario == 'incomplete':
        infmiss = {'perc': [10, 20], #percentage of missing data 
                'type': ['rows', 'random'], #type of missing data 
                'group': [1, 2]} #groups that will have missing values            

    #Make directory to save the results of the experiments          
    res_dir = f'results/2groups/GFA_{args.noise}/{args.K}comps/{args.scenario}'
    if not os.path.exists(res_dir):
            os.makedirs(res_dir)
    for run in range(0, args.num_runs):
        print("Run:", run+1)
        #-GENERATE DATA
        #------------------------------------------------------
        data_file = f'{res_dir}/[{run+1}]Data.dictionary'
        if not os.path.exists(data_file):
            print("Generating data---------")
            if args.scenario == 'complete':
                simData = get_data(args)
            else:
                simData = get_data(args, infmiss)
            #Save file with generated data
            with open(data_file, 'wb') as parameters:
                    pickle.dump(simData, parameters)
            print("Data generated!")            
        else:
            with open(data_file, 'rb') as parameters:
                simData = pickle.load(parameters)
            print("Data loaded!")         

        #-RUN MODEL
        #---------------------------------------------------------------------------------         
        res_file = f'{res_dir}/[{run+1}]ModelOutput.dictionary'
        if not os.path.exists(res_file):  
            print("Running the model---------")
            X_tr = simData['X_tr']
            if 'diagonal' in args.noise:    
                GFAmodel = GFA.IncompleteDataModel(X_tr, args.K)
            else:
                assert args.scenario == 'complete'
                GFAmodel = GFA.OriginalModel(X_tr, args.K)      
            #Fit model
            time_start = time.process_time()
            GFAmodel.fit(X_tr)
            GFAmodel.time_elapsed = time.process_time() - time_start
            print(f'Computational time: {float("{:.2f}".format(GFAmodel.time_elapsed))}s')
            
            #-Predictions (Predict group 2 from group 1) 
            #------------------------------------------------------------------------------
            #Compute mean squared error
            obs_group = np.array([1, 0]) #group 1 was observed 
            gpred = np.where(obs_group == 0)[0][0] #get the non-observed group
            X_test = simData['X_te']
            X_pred = GFAtools(X_test, GFAmodel).PredictView(obs_group, args.noise)
            GFAmodel.MSE = np.mean((X_test[gpred] - X_pred[0]) ** 2)
            #MSE - chance level (MSE between test values and train means)
            Tr_means = np.ones((X_test[gpred].shape[0], X_test[gpred].shape[1])) * \
                np.nanmean(simData['X_tr'][gpred], axis=0)           
            GFAmodel.MSE_chlev = np.mean((X_test[gpred] - Tr_means) ** 2)
            
            #Predict missing values
            if args.scenario == 'incomplete':
                Corr_miss = np.zeros((1,len(infmiss['group'])))
                missing_pred = GFAtools(simData['X_tr'], GFAmodel).PredictMissing(args.num_sources, infmiss)
                missing_true = simData['trueX_miss']
                for i in range(len(infmiss['group'])):
                    Corr_miss[0,i] = np.corrcoef(missing_true[i][missing_true[i] != 0], 
                                        missing_pred[i][missing_pred[i] != 0])[0,1]                   
                GFAmodel.Corr_miss = Corr_miss

            #Save file containing results
            with open(res_file, 'wb') as parameters:
                pickle.dump(GFAmodel, parameters)
        else:
            print('Model already computed!')         

        #Impute median before training the model
        if args.impMedian: 
            assert args.scenario == 'incomplete'
            X_impmed = copy.deepcopy(simData['X_tr'])
            g_miss = np.array(infmiss['group']) - 1 #group with missing values 
            for i in range(g_miss.size):
                for j in range(simData['X_tr'][g_miss[i]].shape[1]):
                    Xtrain_j = simData['X_tr'][g_miss[i]][:,j]
                    X_impmed[i][np.isnan(X_impmed[g_miss[i]][:,j]),j] = np.nanmedian(Xtrain_j)
            
            res_med_file = f'{res_dir}/[{run+1}]ModelOutput_median.dictionary'
            if not os.path.exists(res_med_file): 
                print("Run Model after imp. median----------")
                GFAmodel_median = GFA.OriginalModel(X_impmed, args.K)
                #Fit model
                time_start = time.process_time()
                GFAmodel_median.fit(X_impmed)
                GFAmodel_median.time_elapsed = time.process_time() - time_start
                print(f'Computational time: {float("{:.2f}".format(GFAmodel_median.time_elapsed))}s')
                
                #-Predictions (Predict group 2 from group 1) 
                #------------------------------------------------------------------------------
                obs_group = np.array([1, 0]) #group 1 was observed 
                gpred = np.where(obs_group == 0)[0][0] #get the non-observed group
                X_test = simData['X_te']
                X_pred = GFAtools(X_test, GFAmodel_median).PredictView(obs_group, 'spherical')
                GFAmodel_median.MSE = np.mean((X_test[gpred] - X_pred[0]) ** 2) 
   
                #Save file
                with open(res_med_file, 'wb') as parameters:
                    pickle.dump(GFAmodel_median, parameters)                

    #Plot and save results
    print('Plotting results--------')
    if 'incomplete' in args.scenario:
        vis_simData.main_results(args, res_dir, InfoMiss = infmiss) 
    else:
        vis_simData.main_results(args, res_dir) 
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GFA with two groups")
    parser.add_argument("--scenario", nargs='?', default='incomplete', type=str)
    parser.add_argument("--noise", nargs='?', default='diagonal', type=str)
    parser.add_argument("--num-sources", nargs='?', default=2, type=int)
    parser.add_argument("--K", nargs='?', default=8, type=int)
    parser.add_argument("--num-runs", nargs='?', default=2, type=int)
    parser.add_argument("--impMedian", nargs='?', default=True, type=bool)
    args = parser.parse_args()

    main(args)    


