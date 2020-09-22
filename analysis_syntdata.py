""" Script to run experiments on synthetic data generated 
    with two groups (i.e. two data sources) """

#Author: Fabio S. Ferreira (fabio.ferreira.16@ucl.ac.uk)
#Date: 17 September 2020

import numpy as np
import time
import pickle
import os
import copy
import argparse
import visualization_syntdata
from models import GFA_DiagonalNoiseModel, GFA_OriginalModel
from utils import GFAtools

def get_data(args, infoMiss=None):

    """ 
    Generate synthetic data.

    Parameters
    ----------
    args : local namespace 
        Arguments selected to run the model.

    infoMiss : dict | None, optional.
        Parameters selected to generate data with missing values.  

    Returns
    -------
    data : dict
        Training and test data as well as model parameters used 
        to generate the data.
    
    """
    Ntrain = 400; Ntest = 100
    N = Ntrain + Ntest #  total number of samples
    M = args.num_sources  #number of groups/data sources
    d = np.array([50, 30]) #number of dimensios for each group
    true_K = 4  # true latent components
    # Specify Z manually
    Z = np.zeros((N, true_K))
    for i in range(0, N):
        Z[i,0] = np.sin((i+1)/(N/20))
        Z[i,1] = np.cos((i+1)/(N/20))
        Z[i,2] = 2 * ((i+1)/N-0.5)    
    Z[:,3] = np.random.normal(0, 1, N)          
    # Specify noise precisions manually
    tau = [[] for _ in range(d.size)]
    tau[0] = 5 * np.ones((1,d[0]))[0] 
    tau[1] = 10 * np.ones((1,d[1]))[0]
    # Specify alphas manually
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
        # Get training and test data
        X_train[i] = X[0:Ntrain,:] #Training data
        X_test[i] = X[Ntrain:N,:] #Test data
    #latent variables for training the model    
    Z = Z[0:Ntrain,:]

    # Generate incomplete training data
    if args.scenario == 'incomplete':
        missing_Xtrue = [[] for _ in range(len(infoMiss['group']))]
        for i in range(len(infoMiss['group'])):
            g_miss = infoMiss['group'][i]-1  
            if 'random' in infoMiss['type'][i]: 
                #remove entries randomly
                missing_val =  np.random.choice([0, 1], 
                            size=(X_train[g_miss].shape[0],d[g_miss]), 
                            p=[1-infoMiss['perc'][i-1]/100, infoMiss['perc'][i-1]/100])
                mask_miss =  np.ma.array(X_train[g_miss], mask = missing_val).mask
                missing_Xtrue[i] = np.where(missing_val==1, X_train[g_miss],0)
                X_train[g_miss][mask_miss] = 'NaN'
            elif 'rows' in infoMiss['type'][i]: 
                #remove rows randomly
                missing_Xtrue[i] = np.zeros((Ntrain,d[g_miss]))
                n_rows = int(infoMiss['perc'][i-1]/100 * X_train[g_miss].shape[0])
                shuf_samples = np.arange(Ntrain)
                np.random.shuffle(shuf_samples)
                missing_Xtrue[i][shuf_samples[0:n_rows],:] = X_train[g_miss][shuf_samples[0:n_rows],:]
                X_train[g_miss][shuf_samples[0:n_rows],:] = 'NaN'
    
    # Store data and model parameters            
    data = {'X_tr': X_train, 'X_te': X_test, 'W': W, 'Z': Z, 'tau': tau, 'alpha': alpha, 'true_K': true_K}
    if args.scenario == 'incomplete':
        data.update({'trueX_miss': missing_Xtrue}) 
    return data        

def main(args):
    
    """ 
    Main function to run experiments on synthetic data.

    Parameters
    ----------
    args : local namespace 
        Arguments selected to run the model.
    
    """
    # Define parameters to generate incomplete data sets
    if args.scenario == 'incomplete':
        infmiss = {'perc': [20], #percentage of missing data 
                'type': ['rows'], #type of missing data 
                'group': [2]} #groups that will have missing values            

    # Make directory to save the results of the experiments         
    res_dir = f'results/2groups/GFA_{args.noise}/{args.K}comps/{args.scenario}'
    if not os.path.exists(res_dir):
            os.makedirs(res_dir)
    for run in range(0, args.num_runs):
        print("Run:", run+1)
        
        #-Generate data
        data_file = f'{res_dir}/[{run+1}]Data.dictionary'
        if not os.path.exists(data_file):
            print("Generating data---------")
            if args.scenario == 'complete':
                simData = get_data(args)
            else:
                simData = get_data(args, infmiss)
            #save file with generated data
            with open(data_file, 'wb') as parameters:
                    pickle.dump(simData, parameters)
            print("Data generated!")            
        else:
            with open(data_file, 'rb') as parameters:
                simData = pickle.load(parameters)
            print("Data loaded!")         

        #-Run model        
        res_file = f'{res_dir}/[{run+1}]ModelOutput.dictionary'
        if not os.path.exists(res_file):  
            print("Running the model---------")
            X_tr = simData['X_tr']
            if 'diagonal' in args.noise:    
                GFAmodel = GFA_DiagonalNoiseModel(X_tr, args)
            else:
                assert args.scenario == 'complete'
                GFAmodel = GFA_OriginalModel(X_tr, args)      
            #Fit the model
            time_start = time.process_time()
            GFAmodel.fit(X_tr)
            GFAmodel.time_elapsed = time.process_time() - time_start
            print(f'Computational time: {float("{:.2f}".format(GFAmodel.time_elapsed))}s')
            
            #-Predictions (Predict group 2 from group 1) 
            # Compute mean squared error (MSE)
            obs_group = np.array([1, 0]) #group 1 was observed 
            gpred = np.where(obs_group == 0)[0][0] #get the non-observed group
            X_test = simData['X_te']
            X_pred = GFAtools(X_test, GFAmodel).PredictGroups(obs_group, args.noise)
            GFAmodel.MSE = np.mean((X_test[gpred] - X_pred[0]) ** 2)
            # Compute MSE - chance level (MSE between test values and train means)
            Tr_means = np.ones((X_test[gpred].shape[0], X_test[gpred].shape[1])) * \
                np.nanmean(simData['X_tr'][gpred], axis=0)           
            GFAmodel.MSE_chlev = np.mean((X_test[gpred] - Tr_means) ** 2)
            
            # Predict missing values
            if args.scenario == 'incomplete':
                Corr_miss = np.zeros((1,len(infmiss['group'])))
                missing_pred = GFAtools(simData['X_tr'], GFAmodel).PredictMissing(infmiss)
                missing_true = simData['trueX_miss']
                for i in range(len(infmiss['group'])):
                    Corr_miss[0,i] = np.corrcoef(missing_true[i][missing_true[i] != 0], 
                                        missing_pred[i][missing_pred[i] != 0])[0,1]                   
                GFAmodel.Corr_miss = Corr_miss

            # Save file containing model outputs and predictions
            with open(res_file, 'wb') as parameters:
                pickle.dump(GFAmodel, parameters)
        else:
            print('Model already computed!')         

        # Impute median before training the model
        if args.impMedian: 
            assert args.scenario == 'incomplete'
            X_impmed = copy.deepcopy(simData['X_tr'])
            g_miss = np.array(infmiss['group']) - 1 #group with missing values 
            for i in range(g_miss.size):
                for j in range(simData['X_tr'][g_miss[i]].shape[1]):
                    Xtrain_j = simData['X_tr'][g_miss[i]][:,j]
                    X_impmed[g_miss[i]][np.isnan(X_impmed[g_miss[i]][:,j]),j] = np.nanmedian(Xtrain_j)
            
            res_med_file = f'{res_dir}/[{run+1}]ModelOutput_median.dictionary'
            if not os.path.exists(res_med_file): 
                print("Run Model after imp. median----------")
                GFAmodel_median = GFA_DiagonalNoiseModel(X_impmed, args, imputation=True)
                # Fit the model
                time_start = time.process_time()
                GFAmodel_median.fit(X_impmed)
                GFAmodel_median.time_elapsed = time.process_time() - time_start
                print(f'Computational time: {float("{:.2f}".format(GFAmodel_median.time_elapsed))}s')
                
                # Predictions (Predict group 2 from group 1) 
                obs_group = np.array([1, 0]) #group 1 was observed 
                gpred = np.where(obs_group == 0)[0][0] #get the non-observed group
                X_test = simData['X_te']
                X_pred = GFAtools(X_test, GFAmodel_median).PredictGroups(obs_group, args.noise)
                GFAmodel_median.MSE = np.mean((X_test[gpred] - X_pred[0]) ** 2) 
   
                # Save file containing model outputs and predictions
                with open(res_med_file, 'wb') as parameters:
                    pickle.dump(GFAmodel_median, parameters)                

    # Plot and save results
    print('Plotting results--------')
    if 'incomplete' in args.scenario:
        visualization_syntdata.get_results(args, res_dir, InfoMiss = infmiss) 
    else:
        visualization_syntdata.get_results(args, res_dir) 
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GFA with two groups")
    parser.add_argument("--scenario", nargs='?', default='incomplete', type=str)
    parser.add_argument("--noise", nargs='?', default='diagonal', type=str)
    parser.add_argument("--num-sources", nargs='?', default=2, type=int)
    parser.add_argument("--K", nargs='?', default=6, type=int)
    parser.add_argument("--num-runs", nargs='?', default=1, type=int)
    parser.add_argument("--impMedian", nargs='?', default=True, type=bool)
    args = parser.parse_args()

    main(args)    


