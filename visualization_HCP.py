""" Module to plot and save the results of the experiments 
    on HCP data """

#Author: Fabio S. Ferreira (fabio.ferreira.16@ucl.ac.uk)
#Date: 17 September 2020

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import xlsxwriter
from scipy import io
from utils import GFAtools

def find_relfactors(model, res_dir, BestModel=False):
    
    """ 
    Find the most relevant factors.

    Parameters
    ----------
    model : Outputs of the model.

    res_dir : str
        Path to the directory where the results will be saved.   
    
    BestModel : bool, defaults to False.
        Save results of the best model.

    Returns
    -------
    relfactors_shared : list
        A list of the relevant shared factors.

    relfactors_specific : list
        A list of the relevant factors specific to each data source.
    
    """
    #Calculate explained variance for each factor within
    # data sources
    W = np.concatenate((model.means_w[0], model.means_w[1]), axis=0) 
    ncomps = W.shape[1]; total_var = model.VarExp_total
    d=0; var_within = np.zeros((model.s, ncomps))
    for s in range(model.s):
        Dm = model.d[s]  
        for c in range(ncomps):
            var_within[s,c] = np.sum(W[d:d+Dm,c] ** 2)/total_var * 100
        d += Dm    
    #Calculate relative explained variance for each factor 
    # within data sources
    relvar_within = np.zeros((model.s, ncomps))
    for s in range(model.s):
        for c in range(ncomps):  
            relvar_within[s,c] = var_within[s,c]/np.sum(var_within[s,:]) * 100 

    #Find shared and specific relevant factors
    relfactors_shared = []
    relfactors_specific = [[] for _ in range(model.s)]
    ratio = np.zeros((1, ncomps))
    for c in range(ncomps):
        ratio[0,c] = var_within[1,c]/var_within[0,c]
        if np.any(relvar_within[:,c] > 7.5):
            if ratio[0,c] > 300:
                relfactors_specific[1].append(c)
            elif ratio[0,c] < 0.001:
                relfactors_specific[0].append(c)
            else:
                relfactors_shared.append(c)
    #Save xlsx file with variances, relative variances and ratios
    # of all factors of the best model only      
    if BestModel:
        var_path = f'{res_dir}/Info_factors.xlsx' 
        df = pd.DataFrame({'Factors':range(1, W.shape[1]+1),
                        'Relvar (brain)': list(relvar_within[0,:]), 'Relvar (SMs)': list(relvar_within[1,:]),
                        'Var (brain)': list(var_within[0,:]), 'Var (SMs)': list(var_within[1,:]), 
                        'Ratio (SMs/brain)': list(ratio[0,:])})
        writer = pd.ExcelWriter(var_path, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Sheet1')
        writer.save()                                           

    return relfactors_shared, relfactors_specific                       

def get_results(args, X, ylabels, res_path):

    """ 
    Plot and save the results of the experiments on HCP data.

    Parameters
    ----------
    args : local namespace 
        Arguments selected to run the model.

    X : list 
        List of arrays containing the data matrix of each 
        data source.    
    
    ylabels : array-like
        Array of strings with the labels of the non-imaging
        subject measures.

    res_dir : str
        Path to the directory where the results will be 
        saved.       
    
    """
    nruns = args.num_runs #number of runs
    #initialise variables to save MSEs, correlations and ELBO values
    MSE_beh = np.zeros((nruns, X[1].shape[1]))
    MSE_beh_trmean = np.zeros((nruns, X[1].shape[1]))
    if args.scenario == 'incomplete': 
        Corr_miss = np.zeros((1,nruns))    
    ELBO = np.zeros((1,nruns))
    #initialise file where the results will be written
    ofile = open(f'{res_path}/results.txt','w')   
    for i in range(nruns):  
        print('\nInitialisation: ', i+1, file=ofile)
        print('------------------------------------------------', file=ofile)
        filepath = f'{res_path}[{i+1}]Results.dictionary'
        #Load file containing the model outputs
        with open(filepath, 'rb') as parameters:
            GFA_otp = pickle.load(parameters)  

        print('Computational time (minutes): ', np.round(GFA_otp.time_elapsed/60,2), file=ofile)
        print('Total number of components estimated: ', GFA_otp.k, file=ofile)
        ELBO[0,i] = GFA_otp.L[-1]
        print('ELBO (last value):', np.around(ELBO[0,i],2), file=ofile)

        #-Predictions (predict data source 2 from data source 1)
        #---------------------------------------------------------------------
        # Get training and test sets  
        X_train = [[] for _ in range(GFA_otp.s)]
        X_test = [[] for _ in range(GFA_otp.s)]  
        for s in range(GFA_otp.s):
            X_train[s] = X[s][GFA_otp.indTrain,:]
            if hasattr(GFA_otp, 'X_nan'):
                if np.any(GFA_otp.X_nan[s] == 1):
                    X_train[s][GFA_otp.X_nan[s] == 1] = 'NaN'
            X_test[s] = X[s][GFA_otp.indTest,:]
        # Calculate means of the SMs (non-imaging subject measures) (data source 2)
        Beh_trainmean = np.nanmean(X_train[1], axis=0)               
        # MSE for each SM
        obs_ds = np.array([1, 0]) #data source 1 was observed 
        gpred = np.where(obs_ds == 0)[0][0] #get the non-observed data source  
        X_pred = GFAtools(X_test, GFA_otp).PredictDSources(obs_ds, args.noise)
        for j in range(GFA_otp.d[1]):
            MSE_beh[i,j] = np.mean((X_test[gpred][:,j] - X_pred[0][:,j]) ** 2)/np.mean(X_test[gpred][:,j] ** 2)
            MSE_beh_trmean[i,j] = np.mean((X_test[gpred][:,j] - Beh_trainmean[j]) ** 2)/np.mean(X_test[gpred][:,j] ** 2)
        
        # Predict missing values
        if args.scenario == 'incomplete':
            infmiss = {'perc': [args.pmiss], #percentage of missing data 
                'type': [args.tmiss], #type of missing data 
                'ds': [args.gmiss]} #data sources with missing values          
            miss_pred = GFAtools(X_train, GFA_otp).PredictMissing(infmiss)
            miss_true = GFA_otp.miss_true
            Corr_miss[0,i] = np.corrcoef(miss_true[miss_true != 0], miss_pred[0][miss_pred[0] != 0])[0,1]

        #Calculate total variance explained
        if hasattr(GFA_otp, 'VarExp_total') is False:           
            total_var = 0
            factors_var = 0
            for s in range(GFA_otp.s):
                w = GFA_otp.means_w[s]
                if 'spherical' in args.noise:
                    T = 1/GFA_otp.E_tau[s] * np.identity(w.shape[0])
                else:
                    T = np.diag(1/GFA_otp.E_tau[s][0,:])
                total_var += np.trace(np.dot(w,w.T) + T)
                factors_var += np.trace(np.dot(w,w.T))
            GFA_otp.VarExp_total = total_var
            GFA_otp.VarExp_factors = factors_var
            with open(filepath, 'wb') as parameters:
                pickle.dump(GFA_otp, parameters)        

        #Find the most relevant factors
        relfact_sh, relfact_sp = find_relfactors(GFA_otp, res_path)
        print('Percentage of variance explained by the estimated factors: ', 
                np.around((GFA_otp.VarExp_factors/GFA_otp.VarExp_total) * 100,2), file=ofile)
        print('Relevant shared factors: ', np.array(relfact_sh)+1, file=ofile)
        for m in range(args.num_sources):
            print(f'Relevant specific factors (data source {m+1}): ', np.array(relfact_sp[m])+1, file=ofile)                 
    
    best_ELBO = int(np.argmax(ELBO)+1)
    print('\nOverall results for the best model---------------', file=ofile)  
    print('Best initialisation (ELBO): ', best_ELBO, file=ofile)

    filepath = f'{res_path}[{best_ELBO}]Results.dictionary'
    with open(filepath, 'rb') as parameters:
        GFA_botp = pickle.load(parameters)

    #Plot ELBO of the best model
    L_path = f'{res_path}/ELBO.png'
    plt.figure()
    plt.title('ELBO')
    plt.plot(GFA_botp.L[1:])
    plt.savefig(L_path)
    plt.close()        

    #Find the relevant factors of the best model
    relfact_sh, relfact_sp = find_relfactors(GFA_botp, res_path, BestModel=True)

    #Get brain and SMs factors
    brain_indices = sorted(list(set(relfact_sh + relfact_sp[0]))) 
    SMs_indices = sorted(list(set(relfact_sh + relfact_sp[1])))                                    
    print('Brain factors: ', np.array(brain_indices)+1, file=ofile)
    print('SMs factors: ', np.array(SMs_indices)+1, file=ofile)
    if len(brain_indices) > 0:
        #Save brain factors
        brain_factors = {"wx1": GFA_botp.means_w[0][:,brain_indices]}
        io.savemat(f'{res_path}/wx1.mat', brain_factors)
    if len(SMs_indices) > 0:   
        #Save SMs factors
        sm_factors = {"wx2": GFA_botp.means_w[1][:,SMs_indices]}
        io.savemat(f'{res_path}/wx2.mat', sm_factors)
    #Save relevant latent components
    Z_indices = sorted(list(set(brain_indices + SMs_indices)))
    Z = {"Z": GFA_botp.means_z[:,Z_indices]}
    io.savemat(f'{res_path}/Z.mat', Z)                 

    print(f'\nMulti-output predictions:--------------------------\n', file=ofile)
    sort_beh = np.argsort(np.mean(MSE_beh, axis=0))
    top = 10
    print(f'Top {top} predicted variables: ', file=ofile)
    for l in range(top):
        print(ylabels[sort_beh[l]], file=ofile)
    
    if args.scenario == 'incomplete':
        print('\nPredictions for missing data:--------------------------',file=ofile)
        print(f'Pearsons correlation (avg(std)): {np.around(np.mean(Corr_miss),2)} ({np.around(np.std(Corr_miss),2)})', file=ofile)   

    # Plot MSE of each non-imaging subject measure
    plt.figure(figsize=(10,6))
    pred_path = f'{res_path}/Predictions.png'
    x = np.arange(MSE_beh.shape[1])
    plt.errorbar(x, np.mean(MSE_beh,axis=0), yerr=np.std(MSE_beh,axis=0), fmt='bo', label='Predictions')
    plt.errorbar(x, np.mean(MSE_beh_trmean,axis=0), yerr=np.std(MSE_beh_trmean,axis=0), fmt='yo', label='Train mean')
    plt.legend(loc='upper left',fontsize=17)
    plt.ylim((0.5,1.8))
    plt.title('Prediction of SMs from brain connectivity',fontsize=22)
    plt.xlabel('Non-imaging subject measures',fontsize=19); plt.ylabel('relative MSE',fontsize=19)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14) 
    plt.savefig(pred_path)
    plt.close()
     
    ofile.close()
    print('Visualisation concluded!')       
