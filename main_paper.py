import numpy as np
import numpy.ma as ma
import pickle
import argparse
import time
import os
from scipy import io
from sklearn.preprocessing import StandardScaler
from visualization_paper import results_HCP
from models.GFA import GFA_original, GFA_incomplete

#Settings
def get_args():
    parser = argparse.ArgumentParser()
    #proj_dir = '/cs/research/medic/human-connectome/experiments/fabio_hcp500/data/preproc'
    #proj_dir = '/SAN/medic/human-connectome/experiments/fabio_hcp500/data/preproc'
    proj_dir = '/Users/fabioferreira/Downloads/GFA/data/hcp'
    parser.add_argument('--dir', type=str, default=proj_dir, 
                        help='Main directory')
    parser.add_argument('--noise', type=str, default='PCA', 
                        help='Noise assumption')
    parser.add_argument('--method', type=str, default='GFA', 
                        help='Model to be used')                                       
    parser.add_argument('--k', type=int, default=10,
                        help='number of components to be used')
    parser.add_argument('--n_init', type=int, default=5,
                        help='number of random initializations')
    
    #Preprocessing and training
    parser.add_argument('--standardise', type=bool, default=False, 
                        help='Standardise the data') 
    parser.add_argument('--prediction', type=bool, default=False, 
                        help='Create Train and test sets')
    parser.add_argument('--perc_train', type=int, default=80,
                        help='Percentage of training data')                    

    #Mising data
    parser.add_argument('--remove', type=bool, default=False,
                        help='Remove data')
    parser.add_argument('--perc_miss', type=int, default=1,
                        help='Percentage of missing data')
    parser.add_argument('--type_miss', type=str, default='nonrand',
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

#Creating path
exp_dir = f'{FLAGS.dir}/experiments'
res_dir = f'{exp_dir}/{FLAGS.method}_{FLAGS.noise}/{FLAGS.k}models/{scenario}/{split_data}/'
if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        
#Data
data_dir = f'{FLAGS.dir}/data'
brain_data = io.loadmat(f'{data_dir}/X_pca400.mat') 
clinical_data = io.loadmat(f'{data_dir}/Y.mat')               

#Standardise data
X = [[] for _ in range(2)]
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
            X_train = [[] for _ in range(2)]
            X_test = [[] for _ in range(2)]
            for i in range(2): 
                X_train[i] = X[i][train_ind,:] 
                X_test[i] = X[i][test_ind,:]
        else: 
            X_train = X  

        time_start = time.process_time()
        d = np.array([X_train[0].shape[1], X_train[1].shape[1]])
        if FLAGS.remove:
            if 'random' in FLAGS.type_miss:
                missing =  np.random.choice([0, 1], size=(X_train[FLAGS.vmiss-1].shape[0],d[FLAGS.vmiss-1]), 
                                        p=[1-FLAGS.perc_miss/100, FLAGS.perc_miss/100])
                X_train[FLAGS.vmiss-1][missing == 1] = 'NaN'
            elif 'rows' in FLAGS.type_miss:
                n_rows = int(FLAGS.perc_miss/100 * X_train[FLAGS.vmiss-1].shape[0])
                samples = np.arange(X_train[FLAGS.vmiss-1].shape[0])
                np.random.shuffle(samples)
                X_train[FLAGS.vmiss-1][samples[0:n_rows],:] = 'NaN'
            elif 'nonrand' in FLAGS.type_miss:
                miss_mat = np.zeros((X_train[FLAGS.vmiss-1].shape[0], X_train[FLAGS.vmiss-1].shape[1]))
                miss_mat[X_train[FLAGS.vmiss-1] > FLAGS.perc_miss * np.std(X_train[FLAGS.vmiss-1])] = 1
                miss_mat[X_train[FLAGS.vmiss-1] < - FLAGS.perc_miss * np.std(X_train[FLAGS.vmiss-1])] = 1
                mask_miss =  ma.array(X_train[FLAGS.vmiss-1], mask = miss_mat).mask
                X_train[FLAGS.vmiss-1][mask_miss] = 'NaN'                     
            GFAmodel = GFA_incomplete(X_train, FLAGS.k, d)
        elif 'FA' in FLAGS.noise:   
            GFAmodel = GFA_incomplete(X_train, FLAGS.k, d)
        else:
            GFAmodel = GFA_original(X_train, FLAGS.k, d)
        
        if FLAGS.prediction:
            GFAmodel.X_test = X_test
                    
        L = GFAmodel.fit(X_train)
        GFAmodel.L = L
        GFAmodel.X_train = X_train
        GFAmodel.time_elapsed = (time.process_time() - time_start) 

        with open(filepath, 'wb') as parameters:
            pickle.dump(GFAmodel, parameters)

#visualization
results_HCP(FLAGS.n_init, X[1].shape[1], res_dir)
