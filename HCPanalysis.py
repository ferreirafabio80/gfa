"""Script to run the experiments on HCP data"""

#Author: Fabio S. Ferreira (fabio.ferreira.16@ucl.ac.uk)
#Date: 17 September 2020

import argparse
import time
import os
import pickle
import numpy as np
import pandas as pd
from scipy import io
from sklearn.preprocessing import StandardScaler

import GFA
from utils import GFAtools
from visualization import results_HCP

def get_args():

    """Parses the arguments that are used to run the analysis.
       (the arguments can be input and modified by command-line) 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='results/HCP/1000subjs',
                        help='Project directory')                   
    parser.add_argument('--noise', type=str, default='diagonal', 
                        help='Noise assumption for GFA models') 
    parser.add_argument('--num_sources', type=int, default=2, 
                        help='Number of data sources')                                                          
    parser.add_argument('--K', type=int, default=5,
                        help='number of components to initialised the model')
    parser.add_argument('--num_runs', type=int, default=2,
                        help='number of random initializations (runs)')
    # Preprocessing and training
    parser.add_argument('--standardise', type=bool, default=True, 
                        help='Standardise the data if needed') 
    parser.add_argument('--ptrain', type=int, default=80,
                        help='Percentage of training data')
    parser.add_argument('--scenario', type=str, default='incomplete',
                        help='Data scenario (complete or incomplete)')                                        
    # Missing data info
    # (This is only needed if one wants to simulate how the model predicts 
    # the missing data)
    parser.add_argument('--pmiss', type=int, default=20,
                        help='Percentage of missing data')
    parser.add_argument('--tmiss', type=str, default='rows',
                        help='Type of missing data (completely random values or rows)')
    parser.add_argument('--gmiss', type=int, default=1,
                        help='Data source (group) cointining missing data')                                            

    return parser.parse_args()															                                             

# get arguments 
args = get_args()   
# creating path to save the results of the experiments
exp_dir = f'{args.dir}/experiments'
if args.scenario == 'complete':
    res_dir = f'{exp_dir}/GFA_{args.noise}/{args.K}models/{args.scenario}/training{args.ptrain}/'
else:    
    res_dir = f'{exp_dir}/GFA_{args.noise}/{args.K}models/{args.scenario}/g{args.gmiss}_{args.tmiss}{args.pmiss}_training{args.ptrain}/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# Load data
data_dir = f'{args.dir}/data'
S = args.num_sources #number of data sources
# load preprocessed and deconfounded data matrices (mat files)
brain_data = io.loadmat(f'{data_dir}/X.mat')
clinical_data = io.loadmat(f'{data_dir}/Y.mat') 
# load file containing labels for Y (item-level questionnaire data)
df_ylb = pd.read_excel(f'{data_dir}/LabelsY.xlsx')               
ylabels = df_ylb['Label'].values
# standardise data if needed
X = [[] for _ in range(S)]
X[0] = brain_data['X'][:,0:20]
X[1] = clinical_data['Y'][:,0:20]
if args.standardise:
    X[0] = StandardScaler().fit_transform(X[0])
    X[1] = StandardScaler().fit_transform(X[1])             

for run in range(0, args.num_runs):
    print("Run: ", run+1)
    filepath = f'{res_dir}[{run+1}]Results.dictionary'
    if not os.path.exists(filepath):
        # create an empty file (this is helpful to run multiple 
        #initialization in parallel (e.g. on cluster))
        with open(filepath, 'wb') as parameters:
            pickle.dump(0, parameters)

        # Split data in training and test sets
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
        
        #ensure the training set size is right
        assert round((train_ind.size/N) * 100) == args.ptrain   

        if 'diagonal' in args.noise:
            if args.scenario == 'incomplete':
                if 'random' in args.tmiss:
                    # Remove values randomly from a pre-chosen data source
                    missing =  np.random.choice([0, 1], size=(X_train[args.gmiss-1].shape[0], 
                                X_train[args.gmiss-1].shape[1]), p=[1-args.pmiss/100, args.pmiss/100])
                    mask_miss =  np.ma.array(X_train[args.gmiss-1], mask = missing).mask
                    missing_true = np.where(missing==1, X_train[args.gmiss-1],0)
                    X_train[args.gmiss-1][mask_miss] = 'NaN'                   
                    #ensure the percentage of missing values is correct
                    assert round((mask_miss[mask_miss==1].size/X_train[args.gmiss-1].size) * 100) == args.pmiss
                
                elif 'rows' in args.tmiss:
                    # Remove subjects randomly from a pre-chosen data source
                    n_rows = int(args.pmiss/100 * X_train[args.gmiss-1].shape[0])
                    samples = np.arange(X_train[args.gmiss-1].shape[0])
                    np.random.shuffle(samples)
                    missing_true = X_train[args.gmiss-1][samples[0:n_rows],:]
                    X_train[args.gmiss-1][samples[0:n_rows],:] = 'NaN'
                
                # Initialise the model    
                GFAmodel = GFA.DiagonalNoiseModel(X_train, args)
                #save true missing values
                GFAmodel.miss_true = missing_true    
            else:
                GFAmodel = GFA.DiagonalNoiseModel(X_train, args)    
        else:
            assert args.scenario == 'complete'
            GFAmodel = GFA.OriginalModel(X_train, args)
        
        print("Running the model---------")
        time_start = time.process_time()            
        GFAmodel.fit(X_train)
        GFAmodel.time_elapsed = time.process_time() - time_start
        print(f'Computational time: {float("{:.2f}".format(GFAmodel.time_elapsed/60))} min')
        GFAmodel.indTest = test_ind
        GFAmodel.indTrain = train_ind

        #Save file containing the model outputs
        with open(filepath, 'wb') as parameters:
            pickle.dump(GFAmodel, parameters)        

#visualization
print('Plotting results--------')
results_HCP.get_results(args, X, ylabels, res_dir)

