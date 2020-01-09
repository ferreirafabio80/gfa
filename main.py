import numpy as np
import pandas as pd
from models.GFA_PCA import GFA as GFAcomplete
from models.GFA_FA import GFA as GFAmissing
import pickle
import argparse
import time
import os
#import hdf5storage
from scipy import io
from sklearn.preprocessing import StandardScaler

#Settings
def get_args():
    parser = argparse.ArgumentParser()
    proj_dir = '/cs/research/medic/human-connectome/experiments/fabio_hcp500'
    #proj_dir = '/SAN/medic/human-connectome/experiments/fabio_hcp500'
    parser.add_argument('--dir', type=str, default=proj_dir, 
                        help='Main directory')
    parser.add_argument('--data', type=str, default='data', 
                        help='Dataset')
    parser.add_argument('--type', type=str, default='preproc', 
                        help='Data that will be used')
    parser.add_argument('--noise', type=str, default='FA', 
                        help='Noise assumption')
    parser.add_argument('--method', type=str, default='GFA', 
                        help='Model to be used')                                       
    parser.add_argument('--m', type=int, default=25,
                        help='number of components to be used')
    parser.add_argument('--n_init', type=int, default=1,
                        help='number of random initializations')
    
    #Preprocessing and training
    parser.add_argument('--standardise', type=bool, default=True, 
                        help='Standardise the data') 
    parser.add_argument('--prediction', type=bool, default=False, 
                        help='Create Train and test sets')
    parser.add_argument('--perc_train', type=int, default=80,
                        help='Percentage of training data')                    

    #Mising data
    parser.add_argument('--remove', type=bool, default=True,
                        help='Remove data')
    parser.add_argument('--perc_miss', type=int, default=20,
                        help='Percentage of missing data')
    parser.add_argument('--type_miss', type=str, default='random',
                        help='Type of missing data')
    parser.add_argument('--view_miss', type=str, default='2',
                        help='View with missing data')                                            

    return parser.parse_args()															                                             

FLAGS = get_args()
if FLAGS.remove:
    scenario = f'missing{FLAGS.perc_miss}_{FLAGS.type_miss}_view{FLAGS.view_miss}'
else:
    scenario = f'complete'

if FLAGS.prediction:
    split_data = f'training{FLAGS.perc_train}'
else:
    split_data = 'all'              

#Creating path
res_dir = f'{FLAGS.dir}/{FLAGS.data}/{FLAGS.type}/{FLAGS.method}_{FLAGS.noise}/{FLAGS.m}models/{scenario}/{split_data}/'
if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        
#Data
if 'ABCD' in FLAGS.data:
    data_dir = f'{FLAGS.dir}/{FLAGS.data}/{FLAGS.type}/data'
    if '7500' in FLAGS.type:
    	brain_data = hdf5storage.loadmat(f'{data_dir}/X.mat')
    else:
        brain_data = io.loadmat(f'{data_dir}/X.mat') 
    clinical_data = io.loadmat(f'{data_dir}/Y.mat') 
elif 'ADNI_highD' in FLAGS.data:
    data_dir = f'{FLAGS.dir}/{FLAGS.data}/data' 
    brain_data = io.loadmat(f'{data_dir}/X.mat') 
    clinical_data = io.loadmat(f'{data_dir}/Y_age_gender.mat')
elif 'ADNI_lowD' in FLAGS.data:
    data_dir = f'{FLAGS.dir}/{FLAGS.data}/data'
    brain_data = io.loadmat(f'{data_dir}/X_clean.mat') 
    clinical_data = io.loadmat(f'{data_dir}/Y_splitgender.mat')
elif 'NSPN' in FLAGS.data:
    data_dir = f'{FLAGS.dir}/{FLAGS.data}/{FLAGS.type}/data'
    standardise = False
    brain_data = io.loadmat(f'{data_dir}/Xp.mat') 
    clinical_data = io.loadmat(f'{data_dir}/Yp.mat')
elif 'hcp' in FLAGS.dir:
    data_dir = f'{FLAGS.dir}/{FLAGS.data}/{FLAGS.type}'
    standardise = False
    brain_data = io.loadmat(f'{data_dir}/X.mat') 
    clinical_data = io.loadmat(f'{data_dir}/Y.mat')               

#Standardise data
X = [[] for _ in range(2)]
X[0] = brain_data['X']
X[1] = clinical_data['Y']
if standardise is True:
    X[0] = StandardScaler().fit_transform(X[0])
    X[1] = StandardScaler().fit_transform(X[1])

#Train/test
if FLAGS.prediction is True:
    n_rows = int(FLAGS.perc_train * X[0].shape[0]/100)
    samples = np.arange(X[0].shape[0])
    np.random.shuffle(samples)
    train_ind = samples[0:n_rows]
    test_ind = samples[n_rows:X[0].shape[0]]
    X_train = [[] for _ in range(2)]
    X_test = [[] for _ in range(2)]
    for i in range(2): 
        X_train[i] = X[i][train_ind,:] 
        X_test[i] = X[i][test_ind,:]
else: 
    X_train = X            

GFAmodel = [[] for _ in range(FLAGS.n_init)]
for init in range(0, FLAGS.n_init):
    print("Run:", init+1) 

    time_start = time.process_time()
    d = np.array([X_train[0].shape[1], X_train[1].shape[1]])
    if FLAGS.remove is True:
        if 'random' in FLAGS.type_miss:
            if '1' in FLAGS.view_miss:
                missing =  np.random.choice([0, 1], size=(X_train[0].shape[0],d[0]), 
                                        p=[1-FLAGS.perc_miss/100, FLAGS.perc_miss/100])
                X_train[0][missing == 1] = 'NaN'
            elif '2' in FLAGS.view_miss:
                missing =  np.random.choice([0, 1], size=(X_train[1].shape[0],d[1]), 
                                        p=[1-FLAGS.perc_miss/100, FLAGS.perc_miss/100])
                X_train[1][missing == 1] = 'NaN' 
        
        elif 'rows' in FLAGS.type_miss:
            n_rows = int(FLAGS.perc_miss/100 * X[0].shape[0])
            samples = np.arange(X[0].shape[0])
            np.random.shuffle(samples)
            if '1' in FLAGS.view_miss:
                X_train[0][samples[0:n_rows],:] = 'NaN'
            elif '2' in FLAGS.view_miss:
                X_train[1][samples[0:n_rows],:] = 'NaN'       
        
        GFAmodel[init] = GFAmissing(X_train, FLAGS.m, d)
    elif 'FA' is FLAGS.noise:   
        GFAmodel[init] = GFAmissing(X_train, FLAGS.m, d)
    else:
        GFAmodel[init] = GFAcomplete(X_train, FLAGS.m, d)
    
    if FLAGS.prediction is True:
        GFAmodel[init].X_test = X_test        
    
    L = GFAmodel[init].fit(X_train)
    GFAmodel[init].L = L
    GFAmodel[init].time_elapsed = (time.process_time() - time_start) 

filepath = f'{res_dir}{FLAGS.method}_results.dictionary'
with open(filepath, 'wb') as parameters:

    pickle.dump(GFAmodel, parameters)
