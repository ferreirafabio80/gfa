import numpy as np
import pandas as pd
import GFA as GFA
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
import time
import os
from scipy import io

#Settings
data = 'NSPN'
flag = '2views' 
scenario = 'complete'
noise = 'PCA'
model = 'GFA'
m = 50
directory = f'results/{data}/{flag}/{noise}/{m}models/{scenario}/'
if not os.path.exists(directory):
        os.makedirs(directory)
        
#Data
missing = False
standardise = True
data_dir = f'results/{data}/{flag}/data'
if 'ABCD' in data:
        brain_data = io.loadmat(f'{data_dir}/X.mat') 
        clinical_data = io.loadmat(f'{data_dir}/Y.mat') 
elif 'ADNI_highD' in data: 
        brain_data = io.loadmat(f'{data_dir}/X.mat') 
        clinical_data = io.loadmat(f'{data_dir}/Y_age_gender.mat')
elif 'ADNI_lowD' in data:
        brain_data = io.loadmat(f'{data_dir}/X_clean.mat') 
        clinical_data = io.loadmat(f'{data_dir}/Y_splitgender.mat')
elif 'NSPN' in data:
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

num_init = 1  # number of random initializations
res_BIBFA = [[] for _ in range(num_init)]
for init in range(0, num_init):
    print("Run:", init+1) 

    #removing data from the clinical side
    if missing is True:
        p_miss = 0.20
        missing =  np.random.choice([0, 1], size=(X[1].shape[0],d[1]), p=[1-p_miss, p_miss])
        X[1][missing == 1] = 'NaN'

    time_start = time.process_time()
    d = np.array([X[0].shape[1], X[1].shape[1]])
    res_BIBFA[init] = GFA.BIBFA(X, m, d)
    L = res_BIBFA[init].fit(X)
    res_BIBFA[init].L = L
    res_BIBFA[init].time_elapsed = (time.process_time() - time_start) 

filepath = f'{directory}{model}_results.dictionary'
with open(filepath, 'wb') as parameters:

    pickle.dump(res_BIBFA, parameters)

