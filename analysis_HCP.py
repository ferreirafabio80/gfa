""" Run the experiments on HCP data """

#Author: Fabio S. Ferreira (fabio.ferreira.16@ucl.ac.uk)
#Date: 22 February 2021
import argparse
import time
import os
import pickle
import numpy as np
import pandas as pd
import visualization_HCP 
from scipy import io
from sklearn.preprocessing import StandardScaler
from models import GFA_DiagonalNoiseModel, GFA_OriginalModel
from utils import GFAtools

def compute_mses(X_train, X_test, model):

    """ 
    Calculates the predictions (MSEs) of the non-imaging (NI) 
    measures predicted from brain connectivity

    Parameters
    ----------
    X_train : list
    List of arrays containing the train observations of both groups.

    X_test : list
    List of arrays containing the test observations of both groups.

    model : Outputs of the model.

    Returns
    -------
    MSE_NI_te : array-like
        A row vector with the MSEs calculated between the NI
        measures on the test set and predicted ones.

    MSE_NI_tr : array-like
        A row vector with the MSEs calculated between the NI
        measures on the test set and its train means.
    
    """
    # Calculate means of the NI measures (group 2)
    NI_trmean = np.nanmean(X_train[1], axis=0)               
    # MSE for each NI measure
    obs_ds = np.array([1, 0]) #group 1 was observed 
    gpred = np.where(obs_ds == 0)[0][0] #get the non-observed group  
    X_pred = GFAtools(X_test, model).PredictGroups(obs_ds, args.noise)
    MSE_NI_te = np.zeros((1, model.d[1]))
    MSE_NI_tr = np.zeros((1, model.d[1]))
    for j in range(model.d[1]):
        MSE_NI_te[0,j] = np.mean((X_test[gpred][:,j] - X_pred[0][:,j]) ** 2) / np.mean(X_test[gpred][:,j] ** 2)
        MSE_NI_tr[0,j] = np.mean((X_test[gpred][:,j] - NI_trmean[j]) ** 2) / np.mean(X_test[gpred][:,j] ** 2)

    return MSE_NI_te, MSE_NI_tr   

def main(args): 

    """ 
    Main function to run experiments on HCP data.

    Parameters
    ----------
    args : local namespace 
        Arguments selected to run the model.
    
    """
    # Create path to save the results of the experiments
    exp_dir = f'{args.dir}/experiments'
    if args.scenario == 'complete':
        flag = f'training{args.ptrain}/'
    else:
        flag = f's{args.gmiss}_{args.tmiss}{args.pmiss}_training{args.ptrain}/'    
    res_dir = f'{exp_dir}/GFA_{args.noise}/{args.K}models/{args.scenario}/{flag}'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # Load data
    data_dir = f'{args.dir}/data'
    S = args.num_groups #number of groups
    #load preprocessed and deconfounded data matrices (mat files)
    brain_data = io.loadmat(f'{data_dir}/X.mat')
    clinical_data = io.loadmat(f'{data_dir}/Y.mat') 
    #load file containing labels for Y (item-level questionnaire data)
    df_ylb = pd.read_excel(f'{data_dir}/LabelsY.xlsx')               
    ylabels = df_ylb['Label'].values
    #load matrices
    X = [[] for _ in range(S)]
    X[0] = brain_data['X']
    X[1] = clinical_data['Y']

    for run in range(0, args.num_runs):
        print("Run: ", run+1)
        filepath = f'{res_dir}[{run+1}]Results.dictionary'
        if not os.path.exists(filepath):
            # Create an empty file (this is helpful to run multiple 
            #initializations in parallel (e.g. on cluster))
            with open(filepath, 'wb') as parameters:
                pickle.dump(0, parameters)

            # Randomly split data in training and test sets
            N = X[0].shape[0]
            n_subjs = int(args.ptrain * N/100) #number of subjects for training 
            #randomly shuflle subjects
            samples = np.arange(N)
            np.random.shuffle(samples)
            #get training and test sets
            train_ind = samples[0:n_subjs]
            test_ind = samples[n_subjs:N]
            X_train = [[] for _ in range(S)]
            X_test = [[] for _ in range(S)]
            for i in range(S): 
                X_train[i] = X[i][train_ind,:] 
                X_test[i] = X[i][test_ind,:]

            #standardise data
            if args.scenario == 'complete':
                for i in range(S):
                    scale = StandardScaler().fit(X_train[i])
                    X_train[i] = scale.transform(X_train[i])
                    X_test[i] = scale.transform(X_test[i])    

            #ensure the training set size is right
            assert round((train_ind.size/N) * 100) == args.ptrain   

            params = {'num_groups': args.num_groups,
                      'K': args.K, 'scenario': args.scenario}
            if 'diagonal' in args.noise:
                if args.scenario == 'incomplete':
                    if 'random' in args.tmiss:
                        # Remove values randomly from the group selected
                        missing =  np.random.choice([0, 1], size=(X_train[args.gmiss-1].shape[0], 
                                    X_train[args.gmiss-1].shape[1]), p=[1-args.pmiss/100, args.pmiss/100])
                        mask_miss =  np.ma.array(X_train[args.gmiss-1], mask = missing).mask
                        miss_true = np.where(missing==1, X_train[args.gmiss-1],0)
                        X_train[args.gmiss-1][mask_miss] = 'NaN'                   
                        #ensure the percentage of missing values is correct
                        assert round((mask_miss[mask_miss==1].size/X_train[args.gmiss-1].size) * 100) == args.pmiss
                    
                    elif 'rows' in args.tmiss:
                        # Remove subjects randomly from the group selected
                        n_rows = int(args.pmiss/100 * X_train[args.gmiss-1].shape[0])
                        samples = np.arange(X_train[args.gmiss-1].shape[0])
                        np.random.shuffle(samples)
                        miss_true = X_train[args.gmiss-1][samples[0:n_rows],:]
                        X_train[args.gmiss-1][samples[0:n_rows],:] = 'NaN'
                    
                    # Standardise the data after removing the values
                    for i in range(S):
                        scale = StandardScaler().fit(X_train[i])
                        X_train[i] = scale.transform(X_train[i])
                        X_test[i] = scale.transform(X_test[i])
                    
                    # Initialise the model    
                    GFAmodel = GFA_DiagonalNoiseModel(X_train, params)  
                else:
                    GFAmodel = GFA_DiagonalNoiseModel(X_train, params)    
            else:
                GFAmodel = GFA_OriginalModel(X_train, params)
            
            print("Running the model---------")
            time_start = time.process_time()            
            GFAmodel.fit(X_train)
            GFAmodel.time_elapsed = time.process_time() - time_start
            print(f'Computational time: {float("{:.2f}".format(GFAmodel.time_elapsed/60))} min')

            # Compute MSE for each NI measure
            GFAmodel.MSEs_NI_te, GFAmodel.MSEs_NI_tr = compute_mses(X_train, X_test, GFAmodel)

            # Predict missing values
            if args.scenario == 'incomplete':
                infmiss = {'perc': [args.pmiss], #percentage of missing data 
                    'type': [args.tmiss], #type of missing data 
                    'ds': [args.gmiss]} #groups with missing values          
                miss_pred = GFAtools(X_train, GFAmodel).PredictMissing(infmiss)
                GFAmodel.corrmiss = np.corrcoef(miss_true[miss_true != 0], 
                                miss_pred[0][np.logical_not(np.isnan(miss_pred[0]))])[0,1]
                # Remove mask with NaNs from the model output dictionary
                del GFAmodel.X_nan
            
            # Save file containing the model outputs
            with open(filepath, 'wb') as parameters:
                pickle.dump(GFAmodel, parameters)        

    #visualization
    print('Plotting results--------')
    visualization_HCP.get_results(args, ylabels, res_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GFA using HCP data")
    parser.add_argument('--dir', type=str, default='results/HCP/1000subjs',
                        help='Project directory')                   
    parser.add_argument('--noise', type=str, default='spherical', 
                        help='Noise assumption for GFA models (diagonal or spherical)') 
    parser.add_argument('--num_groups', type=int, default=2, 
                        help='Number of groups')                                                          
    parser.add_argument('--K', type=int, default=80,
                        help='number of factors to initialise the model')
    parser.add_argument('--num_runs', type=int, default=10,
                        help='number of random initializations (runs)')
    parser.add_argument('--ptrain', type=int, default=80,
                        help='Percentage of training data')
    parser.add_argument('--scenario', type=str, default='complete',
                        help='Data scenario (complete or incomplete)')                                        
    # Missing data info
    # (This is only needed if one wants to simulate how the model handles and
    # predicts missing data)
    parser.add_argument('--pmiss', type=int, default=20,
                        help='Percentage of missing data')
    parser.add_argument('--tmiss', type=str, default='random',
                        help='Type of missing data (random values or rows)')
    parser.add_argument('--gmiss', type=int, default=2,
                        help='Group with missing data')
    args = parser.parse_args()

    main(args) 

