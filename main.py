import numpy as np
import numpy.ma as ma
import pickle
import argparse
import time
import os
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import io
from sklearn.preprocessing import StandardScaler
from visualization import results_HCP
from models import GFA
from utils import GFAtools

#Settings
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='results/hcp_paper/1000subjs', #'/SAN/medic/human-connectome/experiments/fabio_hcp1000'
                        help='Main directory')
    parser.add_argument('--nettype', type=str, default='partial', 
                        help='Netmat type (Partial or Full correlation)')                    
    parser.add_argument('--noise', type=str, default='spherical', 
                        help='Noise assumption')
    parser.add_argument('--method', type=str, default='GFA', 
                        help='Model to be used')                                       
    parser.add_argument('--k', type=int, default=150,
                        help='number of components to be used')
    parser.add_argument('--n_init', type=int, default=10,
                        help='number of random initializations')
    
    #Preprocessing and training
    parser.add_argument('--standardise', type=bool, default=True, 
                        help='Standardise the data') 
    parser.add_argument('--perc_train', type=int, default=80,
                        help='Percentage of training data')                    

    #Remove elements from data matrices
    #This is only needed if one wants to simulate how the model predicts the missing data
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

split_data = f'training{FLAGS.perc_train}'  

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
    filepath = f'{res_dir}Results_run{init+1}.dictionary'
    if not os.path.exists(filepath):
        
        #Create an empty file. This is helpful to run the model in parallel on the cluster
        with open(filepath, 'wb') as parameters:
            pickle.dump(0, parameters)

        #Train/test
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

        assert round((train_ind.size/X[0].shape[0]) * 100) == FLAGS.perc_train   

        if 'diagonal' in FLAGS.noise:
            if FLAGS.remove:
                if 'random' in FLAGS.type_miss:
                    missing =  np.random.choice([0, 1], size=(X_train[FLAGS.vmiss-1].shape[0], X_train[FLAGS.vmiss-1].shape[1]), 
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
                
                assert round((mask_miss[mask_miss==1].size/X_train[FLAGS.vmiss-1].size) * 100) == FLAGS.perc_miss    
                GFAmodel = GFA.MissingModel(X_train, FLAGS.k)
                GFAmodel.miss_true = missing_true
                GFAmodel.missing_mask = mask_miss
            else:
                GFAmodel = GFA.MissingModel(X_train, FLAGS.k)    
        else:
            assert FLAGS.remove is False
            GFAmodel = GFA.OriginalModel(X_train, FLAGS.k)
        
        time_start = time.process_time()            
        L = GFAmodel.fit(X_train)
        GFAmodel.time_elapsed = time.process_time() - time_start
        GFAmodel.L = L
        GFAmodel.indTest = test_ind
        GFAmodel.indTrain = train_ind 

        with open(filepath, 'wb') as parameters:
            pickle.dump(GFAmodel, parameters)        

#visualization
best_model, rel_comps, spvar = results_HCP(FLAGS.n_init, X, ylabels, res_dir)

#Run reduced model
ofile = open(f'{res_dir}/reduced_model_{spvar}.txt','w')
X_train = [[] for _ in range(S)]
for i in range(S):
    X_train[i] = X[i][best_model.indTrain,:]
    best_model.means_w[i] = best_model.means_w[i][:,rel_comps]
best_model.means_z = best_model.means_z[:,rel_comps]
if 'spherical' in FLAGS.noise:
    Redmodel = GFA.OriginalModel(X_train, rel_comps.size, lowK_model=best_model)
else:     
    Redmodel = GFA.MissingModel(X_train, rel_comps.size, lowK_model=best_model)
L = Redmodel.fit(X_train)

print(f'Relevant components:', rel_comps, file=ofile)
print(f'Lower bound full model:', best_model.L[-1], file=ofile)
print(f'Lower bound reduced model: ', L[-1], file=ofile)  

#Bayes factor
BF = np.exp(best_model.L[-1]-L[-1]) 
print(f'Bayes factor: ', BF, file=ofile)
ofile.close()

obs_view = np.array([1, 0])
vpred = np.array(np.where(obs_view == 0))
X_test = [[] for _ in range(S)]
for i in range(S):
    X_test[i] = X[i][best_model.indTest,:]
X_pred = GFAtools(X_test, Redmodel, obs_view).PredictView(FLAGS.noise)

#-Metrics
#----------------------------------------------------------------------------------
Beh_trainmean = np.mean(X_train[1], axis=0) 
MSE_trainmean = np.sqrt(np.mean((X_test[vpred[0,0]] - Beh_trainmean) ** 2))
#MSE for each dimension - predict view 2 from view 1
beh_dim = X[1].shape[1]
MSE_beh = np.zeros((1, beh_dim))
MSE_beh_trmean = np.zeros((1, beh_dim))
for j in range(0, beh_dim):
    MSE_beh[0,j] = np.mean((X_test[vpred[0,0]][:,j] - X_pred[:,j]) ** 2)/np.mean(X_test[vpred[0,0]][:,j] ** 2)
    MSE_beh_trmean[0,j] = np.mean((X_test[vpred[0,0]][:,j] - Beh_trainmean[j]) ** 2)/np.mean(X_test[vpred[0,0]][:,j] ** 2)

#Predictions for behaviour
#---------------------------------------------
plt.figure(figsize=(10,8))
pred_path = f'{res_dir}/Predictions_reducedModel_{spvar}.png'
x = np.arange(MSE_beh.shape[1])
plt.errorbar(x, np.mean(MSE_beh,axis=0), yerr=np.std(MSE_beh,axis=0), fmt='bo', label='Predictions')
plt.errorbar(x, np.mean(MSE_beh_trmean,axis=0), yerr=np.std(MSE_beh_trmean,axis=0), fmt='yo', label='Train mean')
plt.legend(loc='upper right',fontsize=14)
plt.ylim((0,2.5))
plt.title('Reduced Model',fontsize=18)
plt.xlabel('Features of view 2',fontsize=16)
plt.ylabel('relative MSE',fontsize=16)
plt.savefig(pred_path)
plt.close()


