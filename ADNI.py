import numpy as np
import pandas as pd
import BIBFA_missing as BCCA
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
import time


#run your code
brain_data = pd.read_csv("ADNI/CT.csv") 
clinical_data = pd.read_csv("ADNI/clinical.csv")
X = [[] for _ in range(2)]
X[0] = brain_data.values
X[1] = clinical_data.values

X[0] = StandardScaler().fit_transform(X[0])
X[1] = StandardScaler().fit_transform(X[1])

d = np.array([X[0].shape[1], X[1].shape[1]])
m = 10

# Incomplete data
#------------------------------------------------------------------------
p_miss = 0.10
for i in range(0,2):
    missing =  np.random.choice([0, 1], size=(X[0].shape[0],d[i]), p=[1-p_miss, p_miss])
    X[i][missing == 1] = 'NaN'

time_start = time.clock()
BCCA = BCCA.BIBFA(X, m, d)
L = BCCA.fit(X)
BCCA.L = L

with open('ADNI/BCCA_ADNI_500subj_missing10.dictionary', 'wb') as parameters:
 
  # Step 3
  pickle.dump(BCCA, parameters)

BCCA.time_elapsed = (time.clock() - time_start)