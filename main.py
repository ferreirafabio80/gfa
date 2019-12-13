import numpy as np
import pandas as pd
from models.GFA_PCA import GFA as GFAcomplete
from models.GFA_FA import GFA as GFAmissing
import pickle
import argparse
import time
import os
import hdf5storage
from scipy import io
from sklearn.preprocessing import StandardScaler

#Settings
def get_args():
    parser = argparse.ArgumentParser()
    proj_dir = 'results'#'/cs/research/medic/human-connectome/experiments/fabio_hcp500'
    parser.add_argument('--dir', type=str, default=proj_dir, 
                        help='Main directory')
    parser.add_argument('--data', type=str, default='ADNI_highD', 
                        help='Dataset')
    parser.add_argument('--type', type=str, default='MMSE', 
                        help='Data that will be used')
    parser.add_argument('--scenario', type=str, default='complete', 
                        help='Including or not missing data')
    parser.add_argument('--noise', type=str, default='PCA', 
                        help='Noise assumption')
    parser.add_argument('--method', type=str, default='GFA', 
                        help='Model to be used')
						
    parser.add_argument('--m', type=int, default=100,
                        help='number of components to be used')
    parser.add_argument('--n_init', type=int, default=1,
                        help='number of random initializations')
    parser.add_argument('--missing', type=int, default=0.2,
                        help='Percentage of missing data')

    return parser.parse_args()															                                             

FLAGS = get_args()

#Creating path
res_dir = f'{FLAGS.dir}/{FLAGS.data}/{FLAGS.type}/{FLAGS.method}_{FLAGS.noise}/{FLAGS.m}models/{FLAGS.scenario}/'
if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        
#Data
missing = False
standardise = True
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

GFAmodel = [[] for _ in range(FLAGS.n_init)]
for init in range(0, FLAGS.n_init):
    print("Run:", init+1) 

    #removing data from the clinical side
    time_start = time.process_time()
    d = np.array([X[0].shape[1], X[1].shape[1]])
    if missing is True:
        missing =  np.random.choice([0, 1], size=(X[1].shape[0],d[1]), 
									p=[1-FLAGS.missing, FLAGS.missing])
        X[1][missing == 1] = 'NaN'
        GFAmodel[init] = GFAmissing(X, FLAGS.m, d)
    elif 'FA' is FLAGS.noise:   
        GFAmodel[init] = GFAmissing(X, FLAGS.m, d)
    else:
        GFAmodel[init] = GFAcomplete(X, FLAGS.m, d)    
    L = GFAmodel[init].fit(X)
    GFAmodel[init].L = L
    GFAmodel[init].time_elapsed = (time.process_time() - time_start) 

filepath = f'{res_dir}{FLAGS.method}_results.dictionary'
with open(filepath, 'wb') as parameters:

    pickle.dump(GFAmodel, parameters)

