import numpy as np
import numpy.ma as ma
import pickle
import argparse
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import io
from sklearn.preprocessing import StandardScaler
from visualization_paper import results_HCP
from models.GFA import GFA_original, GFA_incomplete

#Settings
def get_args():
    parser = argparse.ArgumentParser()
    #proj_dir = '/cs/research/medic/human-connectome/experiments/fabio_hcp500/data/preproc'
    #proj_dir = '/SAN/medic/human-connectome/experiments/fabio_hcp500/data/preproc'
    proj_dir = 'results/hcp_paper/1000subjs'
    parser.add_argument('--dir', type=str, default=proj_dir, 
                        help='Main directory')
    parser.add_argument('--nettype', type=str, default='partial', 
                        help='Netmat type (Partial or Full correlation)')                    
    parser.add_argument('--noise', type=str, default='FA', 
                        help='Noise assumption')
    parser.add_argument('--method', type=str, default='GFA', 
                        help='Model to be used')                                       
    parser.add_argument('--k', type=int, default=40,
                        help='number of components to be used')
    parser.add_argument('--n_init', type=int, default=20,
                        help='number of random initializations')
    
    #Preprocessing and training
    parser.add_argument('--standardise', type=bool, default=True, 
                        help='Standardise the data') 
    parser.add_argument('--prediction', type=bool, default=True, 
                        help='Create Train and test sets')
    parser.add_argument('--perc_train', type=int, default=80,
                        help='Percentage of training data')                    

    #Mising data
    parser.add_argument('--remove', type=bool, default=False,
                        help='Remove data')
    parser.add_argument('--perc_miss', type=int, default=20,
                        help='Percentage of missing data')
    parser.add_argument('--type_miss', type=str, default='random',
                        help='Type of missing data')
    parser.add_argument('--vmiss', type=int, default=2,
                        help='View with missing data')                                            

    return parser.parse_args()															                                             

FLAGS = get_args()
if FLAGS.remove:
    scenario = f'missing{FLAGS.perc_miss}_{FLAGS.type_miss}_view{str(FLAGS.vmiss)}'
else:
    scenario = f'complete'

if FLAGS.prediction:
    split_data = f'training{FLAGS.perc_train}'
else:
    split_data = 'all'  

if 'partial' in FLAGS.nettype:
    net_type = 'partial'
else:
    net_type = 'full'    

#Creating path
exp_dir = f'{FLAGS.dir}/experiments'
res_dir = f'{exp_dir}/{FLAGS.method}_{FLAGS.noise}/{FLAGS.k}models_{net_type}/{scenario}/{split_data}/'
if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        
#Data
data_dir = f'{FLAGS.dir}/data'
S = 2 #number of data sources
if 'partial' in FLAGS.nettype:
    brain_data = io.loadmat(f'{data_dir}/X_par_decnf.mat')
else:
    brain_data = io.loadmat(f'{data_dir}/X_full_decnf.mat') 
clinical_data = io.loadmat(f'{data_dir}/Y_decnf.mat')    
df_ylb = pd.read_excel(f'{data_dir}/LabelsY.xlsx')               
ylabels = df_ylb['Label'].values

#Standardise data
X = [[] for _ in range(S)]
X[0] = brain_data['X']
X[1] = clinical_data['Y']
if FLAGS.standardise:
    X[0] = StandardScaler().fit_transform(X[0])
    X[1] = StandardScaler().fit_transform(X[1])             

print("Run Model------")
for init in range(0, FLAGS.n_init):
    
    print("Run:", init+1)
    #Run model
    filepath = f'{res_dir}GFA_results{init+1}.dictionary'
    if not os.path.exists(filepath):
        dummy = 0
        with open(filepath, 'wb') as parameters:
            pickle.dump(dummy, parameters)

        #Train/test
        if FLAGS.prediction:
            n_rows = int(FLAGS.perc_train * X[0].shape[0]/100)
            samples = np.arange(X[0].shape[0])
            np.random.shuffle(samples)
            train_ind = samples[0:n_rows]
            test_ind = samples[n_rows:X[0].shape[0]]
            X_train = [[] for _ in range(S)]
            X_test = [[] for _ in range(S)]
            for i in range(S): 
                X_train[i] = X[i][train_ind,:] 
                X_test[i] = X[i][test_ind,:]
        else: 
            X_train = X  

        if FLAGS.remove:
            if 'random' in FLAGS.type_miss:
                missing =  np.random.choice([0, 1], size=(X_train[FLAGS.vmiss-1].shape[0],d[FLAGS.vmiss-1]), 
                                        p=[1-FLAGS.perc_miss/100, FLAGS.perc_miss/100])
                mask_miss =  ma.array(X_train[FLAGS.vmiss-1], mask = missing).mask
                missing_true = np.where(missing==1, X_train[FLAGS.vmiss-1],0)
                X_train[FLAGS.vmiss-1][mask_miss] = 'NaN'
            elif 'rows' in FLAGS.type_miss:
                n_rows = int(FLAGS.perc_miss/100 * X_train[FLAGS.vmiss-1].shape[0])
                samples = np.arange(X_train[FLAGS.vmiss-1].shape[0])
                np.random.shuffle(samples)
                missing_true = X_train[FLAGS.vmiss-1][samples[0:n_rows],:]
                X_train[FLAGS.vmiss-1][samples[0:n_rows],:] = 'NaN'
            GFAmodel = GFA_incomplete(X_train, FLAGS.k)
            GFAmodel.miss_true = missing_true
        elif 'FA' in FLAGS.noise:   
            GFAmodel = GFA_incomplete(X_train, FLAGS.k)
        else:
            GFAmodel = GFA_original(X_train, FLAGS.k)
        
        time_start = time.process_time()            
        L = GFAmodel.fit(X_train)
        GFAmodel.L = L
        GFAmodel.time_elapsed = (time.process_time() - time_start)
        if FLAGS.prediction:
            GFAmodel.indTest = test_ind
            GFAmodel.indTrain = train_ind 

        with open(filepath, 'wb') as parameters:
            pickle.dump(GFAmodel, parameters)        

#visualization
best_model, rel_comps = results_HCP(FLAGS.n_init, X, ylabels, res_dir)

#Run reduced model
np.random.seed(42)
ofile = open(f'{res_dir}/reduced_model.txt','w')
X_train = [[] for _ in range(S)]
for i in range(S):
    X_train[i] = X[i][best_model.indTrain,:]
    best_model.means_w[i] = best_model.means_w[i][:,rel_comps]
best_model.means_z = best_model.means_z[:,rel_comps]
if 'PCA' in FLAGS.noise:
    Redmodel = GFA_original(X_train, rel_comps.size, lowK_model=best_model)
else:     
    Redmodel = GFA_incomplete(X_train, rel_comps.size, lowK_model=best_model)
L = Redmodel.fit(X_train)

print(f'Relevant components:', rel_comps, file=ofile)
print(f'Lower bound full model:', best_model.L[-1], file=ofile)
print(f'Lower bound reduced model: ', L[-1], file=ofile)  

#Bayes factor
BF = best_model.L[-1] / L[-1]
print(f'Bayes factor: ', BF, file=ofile)
ofile.close()


