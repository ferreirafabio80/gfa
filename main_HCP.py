import numpy as np
import numpy.ma as ma
import pickle
import argparse
import time
import os
import pandas as pd
import visualization_HCP
import GFA
from scipy import io
from sklearn.preprocessing import StandardScaler
from utils import GFAtools

#Settings
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='results/hcp_paper/1000subjs',
                        help='Project directory')                   
    parser.add_argument('--noise', type=str, default='spherical', 
                        help='Noise assumption for choosing the models') 
    parser.add_argument('--num_sources', type=int, default=2, 
                        help='Number of data sources')                                                          
    parser.add_argument('--K', type=int, default=40,
                        help='number of components to initialised the model')
    parser.add_argument('--n_run', type=int, default=1,
                        help='number of random initializations (runs)')
    # Preprocessing and training
    parser.add_argument('--standardise', type=bool, default=True, 
                        help='Standardise the data if needed') 
    parser.add_argument('--ptrain', type=int, default=80,
                        help='Percentage of training data')                    
    # Missing data info
    # (This is only needed if one wants to simulate how the model predicts 
    # the missing data)
    parser.add_argument('--scenario', type=str, default='complete',
                        help='Data scenario (complete or incomplete)')
    parser.add_argument('--pmiss', type=int, default=20,
                        help='Percentage of missing data')
    parser.add_argument('--tmiss', type=str, default='rows',
                        help='Type of missing data (completely random values or rows)')
    parser.add_argument('--smiss', type=int, default=1,
                        help='Data source cointining missing data')                                            

    return parser.parse_args()															                                             

args = get_args()   
# Creating path to save the results of the experiments
exp_dir = f'{args.dir}/experiments'
res_dir = f'{exp_dir}/GFA_{args.noise}/{args.K}models/{args.scenario}/training{args.ptrain}/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# Data
data_dir = f'{args.dir}/data'
S = args.num_sources #number of data sources
#load preprocessed and deconfounded data matrices (mat files)
brain_data = io.loadmat(f'{data_dir}/X.mat')
clinical_data = io.loadmat(f'{data_dir}/Y.mat') 
#load file containing labels for Y (item-level questionnaire data)
df_ylb = pd.read_excel(f'{data_dir}/LabelsY.xlsx')               
ylabels = df_ylb['Label'].values
#standardise data if needed
X = [[] for _ in range(S)]
X[0] = brain_data['X']
X[1] = clinical_data['Y']
if args.standardise:
    X[0] = StandardScaler().fit_transform(X[0])
    X[1] = StandardScaler().fit_transform(X[1])             

for run in range(0, args.n_run):
    
    print("Run: ", run+1)
    filepath = f'{res_dir}[{run+1}]Results.dictionary'
    if not os.path.exists(filepath):
        #Create an empty file. This is helpful to run the model in parallel 
        # on clusters
        with open(filepath, 'wb') as parameters:
            pickle.dump(0, parameters)

        #Split data in training and test sets
        N = X[0].shape[0]
        n_subjs = int(args.ptrain * N/100) #number of subjects for training 
        #randomly shuflle subjects
        samples = np.arange(N)
        np.random.shuffle(samples)
        train_ind = samples[0:n_subjs]
        test_ind = samples[n_subjs:N]
        X_train = [[] for _ in range(S)]
        X_test = [[] for _ in range(S)]
        for i in range(S): 
            X_train[i] = X[i][train_ind,:] 
            X_test[i] = X[i][test_ind,:]
        #make sure the training set size is correct
        assert round((train_ind.size/N) * 100) == args.ptrain   

        if 'diagonal' in args.noise:
            if args.scenario == 'incomplete':
                if 'random' in args.tmiss:
                    #Remove values randomly from a pre-chosen data source
                    missing =  np.random.choice([0, 1], size=(X_train[args.smiss-1].shape[0], 
                                X_train[args.smiss-1].shape[1]), p=[1-args.pmiss/100, args.pmiss/100])
                    mask_miss =  ma.array(X_train[args.smiss-1], mask = missing).mask
                    missing_true = np.where(missing==1, X_train[args.smiss-1],0)
                    X_train[args.smiss-1][mask_miss] = 'NaN'
                    #make sure the percentage of missing values is correct
                    assert round((mask_miss[mask_miss==1].size/X_train[args.smiss-1].size) * 100) == args.pmiss
                    #initialise the model
                    GFAmodel = GFA.DiagonalNoiseModel(X_train, args.k)
                    #save true missing values and mask for NANs 
                    GFAmodel.miss_true = missing_true
                    GFAmodel.missing_mask = mask_miss
                elif 'rows' in args.type_miss:
                    #Remove subjects randomly from a pre-chosen data source
                    n_rows = int(args.pmiss/100 * X_train[args.smiss-1].shape[0])
                    samples = np.arange(X_train[args.smiss-1].shape[0])
                    np.random.shuffle(samples)
                    missing_true = X_train[args.smiss-1][samples[0:n_rows],:]
                    X_train[args.smiss-1][samples[0:n_rows],:] = 'NaN'
                    #initialise the model    
                    GFAmodel = GFA.DiagonalNoiseModel(X_train, args.k)
                    #save true missing values and mask for NANs
                    GFAmodel.miss_true = missing_true
                    GFAmodel.missing_rows = samples[0:n_rows]
            else:
                GFAmodel = GFA.DiagonalNoiseModel(X_train, args.k)    
        else:
            assert args.scenario == 'complete'
            GFAmodel = GFA.OriginalModel(X_train, args)
        
        print("Running the model---------")
        time_start = time.process_time()            
        GFAmodel.fit(X_train)
        GFAmodel.time_elapsed = time.process_time() - time_start
        print(f'Computational time: {float("{:.2f}".format(GFAmodel.time_elapsed/60))} min')
        #Save train and test indices to be used in the visualization script 
        GFAmodel.indTest = test_ind
        GFAmodel.indTrain = train_ind

        #Save file containing the model outputs
        with open(filepath, 'wb') as parameters:
            pickle.dump(GFAmodel, parameters)        

#visualization
print('Plotting results--------')
visualization_HCP.main_results(args.n_run, X, ylabels, res_dir)

