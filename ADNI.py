import numpy as np
import pandas as pd
import GFA_fact
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
import time

#run your code
brain_data = pd.read_csv("results/ADNI/CT.csv") 
clinical_data = pd.read_csv("results/ADNI/clinical.csv")
X = [[] for _ in range(2)]
X[0] = brain_data.values
X[1] = clinical_data.values

X[0] = StandardScaler().fit_transform(X[0])
X[1] = StandardScaler().fit_transform(X[1])

d = np.array([X[0].shape[1], X[1].shape[1]])
m = 15

num_init = 10  # number of random initializations
res_BIBFA = [[] for _ in range(num_init)]
for init in range(0, num_init):
    print("Run:", init+1)
    
    # Incomplete data
    #------------------------------------------------------------------------
    """ p_miss = 0.10
    #for i in range(0,2):
    #    missing =  np.random.choice([0, 1], size=(X[0].shape[0],d[i]), p=[1-p_miss, p_miss])
    #    X[i][missing == 1] = 'NaN' 

    #removing data from the clinical side
    missing =  np.random.choice([0, 1], size=(X[1].shape[0],d[1]), p=[1-p_miss, p_miss])
    X[1][missing == 1] = 'NaN' """

    time_start = time.process_time()
    res_BIBFA[init] = GFA_fact.BIBFA(X, m, d)
    L = res_BIBFA[init].fit(X)
    res_BIBFA[init].L = L
    res_BIBFA[init].time_elapsed = (time.process_time() - time_start)

name_file = f'results/ADNI/BIBFA_FAcomplete.dictionary'
with open(name_file, 'wb') as parameters:
    
    pickle.dump(res_BIBFA, parameters)

