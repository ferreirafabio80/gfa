""" Module to plot and save the results of the experiments 
    on HCP data """

#Author: Fabio S. Ferreira (fabio.ferreira.16@ucl.ac.uk)
#Date: 22 February 2021
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import xlsxwriter
import os
import sys
from scipy import io

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
        A list of the relevant factors specific to each group.
    
    """
    #Calculate explained variance for each factor within groups
    W = np.concatenate((model.means_w[0], model.means_w[1]), axis=0) 
    ncomps = W.shape[1]; total_var = model.VarExp_total
    d=0; var_within = np.zeros((model.s, ncomps))
    for s in range(model.s):
        Dm = model.d[s]  
        for c in range(ncomps):
            var_within[s,c] = np.sum(W[d:d+Dm,c] ** 2)/total_var * 100
        d += Dm    
    #Calculate relative explained variance for each factor within groups
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
                        'Relvar (brain)': list(relvar_within[0,:]), 'Relvar (NI measures)': list(relvar_within[1,:]),
                        'Var (brain)': list(var_within[0,:]), 'Var (NI measures)': list(var_within[1,:]), 
                        'Ratio (NI/brain)': list(ratio[0,:])})
        writer = pd.ExcelWriter(var_path, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Sheet1')
        writer.save()                                           

    return relfactors_shared, relfactors_specific                       

def get_results(args, ylabels, res_path):

    """ 
    Plot and save the results of the experiments on HCP data.

    Parameters
    ----------
    args : local namespace 
        Arguments selected to run the model.
    
    ylabels : array-like
        Array of strings with the labels of the non-imaging
        subject measures.

    res_dir : str
        Path to the directory where the results will be 
        saved.       
    
    """
    nruns = args.num_runs #number of runs
    #initialise variables to save MSEs, correlations and ELBO values
    MSE_NI_te = np.zeros((nruns, ylabels.size))
    MSE_NI_tr = np.zeros((nruns, ylabels.size))
    if args.scenario == 'incomplete': 
        Corr_miss = np.zeros((1,nruns))    
    ELBO = np.zeros((1,nruns))
    #initialise file where the results will be written
    ofile = open(f'{res_path}/results.txt','w')   
    for i in range(nruns):  
        print('\nInitialisation: ', i+1, file=ofile)
        print('------------------------------------------------', file=ofile)
        filepath = f'{res_path}[{i+1}]Results.dictionary'
        #ensure file is not empty
        assert os.stat(filepath).st_size > 5
        #Load file containing the model outputs
        with open(filepath, 'rb') as parameters:
            GFA_otp = pickle.load(parameters)  

        print('Computational time (minutes): ', np.round(GFA_otp.time_elapsed/60,2), file=ofile)
        print('Total number of factors estimated: ', GFA_otp.k, file=ofile)
        ELBO[0,i] = GFA_otp.L[-1]
        print('ELBO (last value):', np.around(ELBO[0,i],2), file=ofile)

        # Get predictions (predict NI measures from brain connectivity)
        #MSE_NI_te[i,:] = GFA_otp.MSEs_NI_te
        #MSE_NI_tr[i,:] = GFA_otp.MSEs_NI_tr
        
        # Get predictions (missing values)
        if args.scenario == 'incomplete':
            Corr_miss[0,i] = GFA_otp.corrmiss

        #Calculate total variance explained
        if hasattr(GFA_otp, 'VarExp_total') is False:           
            total_var = 0
            factors_var = 0
            for s in range(GFA_otp.s):
                w = GFA_otp.means_w[s]
                if 'spherical' in args.noise:
                    T = 1/GFA_otp.E_tau[0,s] * np.identity(w.shape[0])
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
        for m in range(args.num_groups):
            print(f'Relevant specific factors (group {m+1}): ', np.array(relfact_sp[m])+1, file=ofile)                       
    
    best_ELBO = int(np.argmax(ELBO)+1)
    print('\nOverall results for the best model', file=ofile)  
    print('------------------------------------------------', file=ofile)
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

    #Get brain and NI factors
    brain_indices = sorted(list(set(relfact_sh + relfact_sp[0]))) 
    NI_indices = sorted(list(set(relfact_sh + relfact_sp[1])))                                    
    print('Brain factors: ', np.array(brain_indices)+1, file=ofile)
    print('NI factors: ', np.array(NI_indices)+1, file=ofile)
    if len(brain_indices) > 0:
        #Save brain factors
        brain_factors = {"wx1": GFA_botp.means_w[0][:,brain_indices]}
        io.savemat(f'{res_path}/wx1.mat', brain_factors)
    if len(NI_indices) > 0:   
        #Save NI factors
        NI_factors = {"wx2": GFA_botp.means_w[1][:,NI_indices]}
        io.savemat(f'{res_path}/wx2.mat', NI_factors)
    #Save relevant latent factors
    Z_indices = sorted(list(set(brain_indices + NI_indices)))
    Z = {"Z": GFA_botp.means_z[:,Z_indices]}
    io.savemat(f'{res_path}/Z.mat', Z)                 

    print(f'\nMulti-output predictions:', file=ofile)
    print('------------------------------------------------', file=ofile)
    sort_beh = np.argsort(np.mean(MSE_NI_te, axis=0))
    top = 10
    print(f'Top {top} predicted variables: ', file=ofile)
    for l in range(top):
        print(ylabels[sort_beh[l]], file=ofile)
    
    if args.scenario == 'incomplete':
        print('\nPredictions for missing data:',file=ofile)
        print('------------------------------------------------', file=ofile)
        print(f'Pearsons correlation (avg(std)): {np.around(np.mean(Corr_miss),3)} ({np.around(np.std(Corr_miss),3)})', file=ofile)   

    # Plot MSE of each non-imaging subject measure
    plt.figure(figsize=(10,6))
    pred_path = f'{res_path}/Predictions.png'
    x = np.arange(MSE_NI_te.shape[1])
    plt.errorbar(x, np.mean(MSE_NI_te,axis=0), yerr=np.std(MSE_NI_te,axis=0), fmt='bo', label='Predictions')
    plt.errorbar(x, np.mean(MSE_NI_tr,axis=0), yerr=np.std(MSE_NI_tr,axis=0), fmt='yo', label='Train mean')
    plt.legend(loc='upper left',fontsize=17)
    plt.ylim((np.min(MSE_NI_te)-0.2, np.max(MSE_NI_te)+0.1))
    plt.title('Prediction of NI measures from brain connectivity',fontsize=22)
    plt.xlabel('Non-imaging subject measures',fontsize=19); plt.ylabel('relative MSE',fontsize=19)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14) 
    plt.savefig(pred_path)
    plt.close()
     
    ofile.close()
    print('Visualisation concluded!')

    return GFA_botp, Z_indices      
