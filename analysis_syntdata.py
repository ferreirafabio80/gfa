""" Run experiments on synthetic data """

#Author: Fabio S. Ferreira (fabio.ferreira.16@ucl.ac.uk)
#Date: 22 February 2021
import numpy as np
import time
import pickle
import os
import copy
import argparse
import visualization_syntdata
from models import GFA_DiagonalNoiseModel, GFA_OriginalModel
from utils import GFAtools

def generate_missdata(X_train, infoMiss):
    """ 
    Generate missing data in the training data.

    Parameters
    ----------
    X_train : list 
        List of arrays containing the data matrix of each group.

    infoMiss : dict 
        Parameters to generate data with missing values.  

    Returns
    -------
    X_train : list 
        List of arrays containing the training data. The groups
        specified in infoMiss will have missing values.

    missing_Xtrue : list 
        List of arrays containing the true values removed from the
        groups selected in infoMiss.     
    
    """
    missing_Xtrue = [[] for _ in range(len(infoMiss['ds']))]
    for i in range(len(infoMiss['ds'])):
        g_miss = infoMiss['ds'][i]-1  
        if 'random' in infoMiss['type'][i]: 
            #remove entries randomly
            missing_val =  np.random.choice([0, 1], 
                        size=(X_train[g_miss].shape[0],X_train[g_miss].shape[1]), 
                        p=[1-infoMiss['perc'][i]/100, infoMiss['perc'][i]/100])
            mask_miss =  np.ma.array(X_train[g_miss], mask = missing_val).mask
            missing_Xtrue[i] = np.where(missing_val==1, X_train[g_miss],0)
            X_train[g_miss][mask_miss] = 'NaN'
        elif 'rows' in infoMiss['type'][i]: 
            #remove rows randomly
            Ntrain = X_train[g_miss].shape[0]
            missing_Xtrue[i] = np.zeros((Ntrain, X_train[g_miss].shape[1]))
            n_rows = int(infoMiss['perc'][i]/100 * Ntrain)
            shuf_samples = np.arange(Ntrain)
            np.random.shuffle(shuf_samples)
            missing_Xtrue[i][shuf_samples[0:n_rows],:] = X_train[g_miss][shuf_samples[0:n_rows],:]
            X_train[g_miss][shuf_samples[0:n_rows],:] = 'NaN'
        elif 'nonrand' in infoMiss['type'][i]:
            miss_mat = np.zeros((X_train[g_miss].shape[0], X_train[g_miss].shape[1]))
            miss_mat[X_train[g_miss] > infoMiss['perc'][i] * np.std(X_train[g_miss])] = 1
            miss_mat[X_train[g_miss] < - infoMiss['perc'][i] * np.std(X_train[g_miss])] = 1
            mask_miss =  np.ma.array(X_train[g_miss], mask = miss_mat).mask
            missing_Xtrue[i] = np.where(miss_mat==1, X_train[g_miss],0)
            X_train[g_miss][mask_miss] = 'NaN'    
    return X_train, missing_Xtrue 

def get_data_2g(args, infoMiss=None):

    """ 
    Generate synthetic data with 2 groups.

    Parameters
    ----------
    args : local namespace 
        Arguments selected to run the model.

    infoMiss : dict | None, optional.
        Parameters to generate data with missing values.  

    Returns
    -------
    data : dict
        Training and test data as well as model parameters used 
        to generate the data.
    
    """
    Ntrain = 400; Ntest = 100
    N = Ntrain + Ntest #  total number of samples
    M = args.num_groups  #number of groups
    d = np.array([50, 30]) #number of dimensios in each group
    true_K = 4  # true latent factors
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
        X_train, missing_Xtrue = generate_missdata(X_train, infoMiss)        
    
    # Store data and model parameters            
    data = {'X_tr': X_train, 'X_te': X_test, 'W': W, 'Z': Z, 
            'tau': tau, 'alpha': alpha, 'true_K': true_K}
    if args.scenario == 'incomplete':
        # Save true missing values
        data.update({'trueX_miss': missing_Xtrue}) 
    return data               

def get_data_3g(args, infoMiss=None):

    """ 
    Generate synthetic data with 3 groups.

    Parameters
    ----------
    args : local namespace 
        Arguments selected to run the model.

    infoMiss : dict | None, optional.
        Parameters selected to generate data with missing values.  

    Returns
    -------
    data : dict
        Training and test data, as well as model parameters used 
        to generate the data.
    
    """
    Ntrain = 400; Ntest = 100
    N = Ntrain + Ntest #  total number of samples
    M = args.num_groups  #number of groups
    d = np.array([50, 30, 20]) #number of dimensios in each group
    true_K = 4  # true latent factors
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
    tau[2] = 8 * np.ones((1,d[2]))[0]
    # Specify alphas manually
    alpha = np.zeros((M, true_K))
    alpha[0,:] = np.array([1,1,1e6,1])
    alpha[1,:] = np.array([1,1,1,1e6]) 
    alpha[2,:] = np.array([1,1e6,1,1e6]) 
    
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
        X_train, missing_Xtrue = generate_missdata(X_train, infoMiss)
    
    # Store data and model parameters            
    data = {'X_tr': X_train, 'X_te': X_test, 'W': W, 'Z': Z, 'tau': tau, 
            'alpha': alpha, 'true_K': true_K}
    if args.scenario == 'incomplete':
        # Save true missing values
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
        infmiss = {'perc': [80], #percentage of missing data
                'type': ['random'], #type of missing data 
                'ds': [2]} #groups that will have missing values            

    # Make directory to save the results of the experiments
    if 'incomplete' in args.scenario:
        ds = infmiss.get('ds'); type = infmiss.get('type'); p = infmiss.get('perc')
        flag = f's{ds}_{type}{p}/'
    else:
        flag = ''         
    #res_dir = f'results/{args.num_groups}groups/GFA_{args.noise}/{args.K}comps/{args.scenario}/{flag}'
    res_dir = f'results/revisions_exp/GFA_{args.noise}/{args.K}comps/{args.scenario}/{flag}'
    if not os.path.exists(res_dir):
            os.makedirs(res_dir)
    for run in range(0, args.num_runs):
        print('------------')
        print("Run:", run+1)
        
        # Generate data
        data_file = f'{res_dir}/[{run+1}]Data.dictionary'
        if not os.path.exists(data_file):
            print("Generating data---------")
            if args.scenario == 'complete':
                if args.num_groups == 2:
                    synt_data = get_data_2g(args)
                elif args.num_groups == 3:
                    synt_data = get_data_3g(args)    
            else:
                if args.num_groups == 2:
                    synt_data = get_data_2g(args,infmiss)
                elif args.num_groups == 3:
                    synt_data = get_data_3g(args,infmiss) 
            #save file with generated data
            with open(data_file, 'wb') as parameters:
                    pickle.dump(synt_data, parameters)
            print("Data generated!")            
        else:
            with open(data_file, 'rb') as parameters:
                synt_data = pickle.load(parameters)
            print("Data loaded!")      

        # Run model        
        res_file = f'{res_dir}/[{run+1}]ModelOutput.dictionary'
        if not os.path.exists(res_file):  
            print("Running the model---------")
            X_tr = synt_data['X_tr']
            params = {'num_groups': args.num_groups,
                      'K': args.K, 'scenario': args.scenario}
            if 'diagonal' in args.noise:    
                GFAmodel = GFA_DiagonalNoiseModel(X_tr, params)
            else:
                # ensure there are no missing values
                assert params['scenario'] == 'complete'
                GFAmodel = GFA_OriginalModel(X_tr, params)      
            #Fit the model
            time_start = time.process_time()
            GFAmodel.fit(X_tr)
            GFAmodel.time_elapsed = time.process_time() - time_start
            print(f'Computational time: {float("{:.2f}".format(GFAmodel.time_elapsed))}s')
            
            # Predictions
            # Compute mean squared error (MSE)
            if args.num_groups == 2:
                obs_ds = np.array([0, 1]) #group 1 was observed
            elif args.num_groups == 3:
                obs_ds = np.array([1, 1, 0]) #group 1 and 2 were observed
            gpred = np.where(obs_ds == 0)[0][0] #get the non-observed group
            X_test = synt_data['X_te']
            X_pred = GFAtools(X_test, GFAmodel).PredictGroups(obs_ds, args.noise)
            GFAmodel.MSE = np.mean((X_test[gpred] - X_pred[0]) ** 2)
            
            # Compute MSE - chance level (MSE between test values and train means)
            Tr_means = np.ones((X_test[gpred].shape[0], X_test[gpred].shape[1])) * \
                np.nanmean(synt_data['X_tr'][gpred], axis=0)           
            GFAmodel.MSE_chlev = np.mean((X_test[gpred] - Tr_means) ** 2)
            
            # Predict missing values
            if args.scenario == 'incomplete':
                Corr_miss = np.zeros((1,len(infmiss['ds'])))
                missing_pred = GFAtools(synt_data['X_tr'], GFAmodel).PredictMissing(infmiss)
                missing_true = synt_data['trueX_miss']
                for i in range(len(infmiss['ds'])):
                    Corr_miss[0,i] = np.corrcoef(missing_true[i][missing_true[i] != 0], 
                                        missing_pred[i][np.logical_not(np.isnan(missing_pred[i]))])[0,1]                   
                GFAmodel.Corr_miss = Corr_miss

            # Save file containing model outputs and predictions
            with open(res_file, 'wb') as parameters:
                pickle.dump(GFAmodel, parameters)
        else:
            print('Model already computed!')         

        # Impute median before training the model
        if args.impMedian: 
            #ensure scenario and noise were correctly selected
            assert args.scenario == 'incomplete' and args.noise == 'diagonal'
            X_impmed = copy.deepcopy(synt_data['X_tr'])
            g_miss = np.array(infmiss['ds']) - 1 #group with missing values 
            for i in range(g_miss.size):
                for j in range(synt_data['X_tr'][g_miss[i]].shape[1]):
                    Xtrain_j = synt_data['X_tr'][g_miss[i]][:,j]
                    X_impmed[g_miss[i]][np.isnan(X_impmed[g_miss[i]][:,j]),j] = np.nanmedian(Xtrain_j)
            
            res_med_file = f'{res_dir}/[{run+1}]ModelOutput_median.dictionary'
            if not os.path.exists(res_med_file): 
                print("Run Model after imp. median----------")
                params = {'num_groups': args.num_groups,
                      'K': args.K, 'scenario': args.scenario}

                noise = 'spherical'
                if 'spherical' in noise:
                    GFAmodel_median = GFA_OriginalModel(X_impmed, params)
                else:
                    GFAmodel_median = GFA_DiagonalNoiseModel(X_impmed, params, imputation=True)    
                # Fit the model
                time_start = time.process_time()
                GFAmodel_median.fit(X_impmed)
                GFAmodel_median.time_elapsed = time.process_time() - time_start
                print(f'Computational time: {float("{:.2f}".format(GFAmodel_median.time_elapsed))}s')
                
                # Predictions 
                if args.num_groups == 2:
                    obs_ds = np.array([1, 0]) #group 1 was observed
                elif args.num_groups == 3:
                    obs_ds = np.array([1, 1, 0]) #group 1 and 2 were observed
                gpred = np.where(obs_ds == 0)[0][0] #get the non-observed group
                X_test = synt_data['X_te']
                X_pred = GFAtools(X_test, GFAmodel_median).PredictGroups(obs_ds, noise)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", nargs='?', default='complete', type=str,
                        help='Data scenario (complete or incomplete)')
    parser.add_argument("--noise", nargs='?', default='diagonal', type=str,
                        help='Noise assumption for GFA models (diagonal or spherical)')
    parser.add_argument("--num-groups", nargs='?', default=2, type=int,
                        help='Number of groups')
    parser.add_argument("--K", nargs='?', default=15, type=int,
                        help='number of factors to initialise the model')
    parser.add_argument("--num-runs", nargs='?', default=10, type=int,
                        help='number of random initializations (runs)')
    parser.add_argument("--impMedian", nargs='?', default=False, type=bool,
                        help='(not) impute median')
    args = parser.parse_args()

    main(args)    


