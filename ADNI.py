import numpy as np
import pandas as pd
import GFA_fact as GFA
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
import time
import os
from scipy import io

""" #ADNI - low dimensional
brain_data = pd.read_csv("results/ADNI/CT.csv") 
clinical_data = pd.read_csv("results/ADNI/clinical_overall.csv")
X = [[] for _ in range(2)]
X[0] = brain_data.values
X[1] = clinical_data.values """

#ADNI - joao's mat files
brain_data = io.loadmat("results/ADNI_joao/X_clean.mat") 
clinical_data = io.loadmat("results/ADNI_joao/Y_new.mat") 
X = [[] for _ in range(2)]
X[0] = brain_data['X']
X[1] = clinical_data['Y']

X[0] = StandardScaler().fit_transform(X[0])
X[1] = StandardScaler().fit_transform(X[1])

d = np.array([X[0].shape[1], X[1].shape[1]])
m = 15
num_init = 3  # number of random initializations
res_BIBFA = [[] for _ in range(num_init)]
for init in range(0, num_init):
    print("Run:", init+1)
    
    # Incomplete data
    #------------------------------------------------------------------------
    p_miss = 0.20
    #for i in range(0,2):
    #    missing =  np.random.choice([0, 1], size=(X[0].shape[0],d[i]), p=[1-p_miss, p_miss])
    #    X[i][missing == 1] = 'NaN' 

    #removing data from the clinical side
    missing =  np.random.choice([0, 1], size=(X[1].shape[0],d[1]), p=[1-p_miss, p_miss])
    X[1][missing == 1] = 'NaN'

    time_start = time.process_time()
    res_BIBFA[init] = GFA.BIBFA(X, m, d)
    L = res_BIBFA[init].fit(X)
    res_BIBFA[init].L = L
    res_BIBFA[init].time_elapsed = (time.process_time() - time_start) 

data = 'ADNI_joao/overall_scores_gender_brainclean' 
scenario = 'missing20'
noise = 'FA'
model = 'GFA'
directory = f'results/{data}/{noise}/{m}models/{scenario}/'
filepath = f'{directory}{model}_results.dictionary'
if not os.path.exists(directory):
        os.makedirs(directory)

with open(filepath, 'wb') as parameters:

    pickle.dump(res_BIBFA, parameters)

