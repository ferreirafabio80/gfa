import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import copy
import os
from scipy import io
from utils import GFAtools
from sklearn.metrics.pairwise import cosine_similarity

def compute_variances(W, d, total_var, spvar, res_path, BestModel = False):

    #Explained variance
    var1 = np.zeros((1, W.shape[1])) 
    var2 = np.zeros((1, W.shape[1])) 
    var = np.zeros((1, W.shape[1]))
    ratio = np.zeros((1, W.shape[1]))
    for c in range(0, W.shape[1]):
        w = np.reshape(W[:,c],(W.shape[0],1))
        w1 = np.reshape(W[0:d[0],c],(d[0],1))
        w2 = np.reshape(W[d[0]:d[0]+d[1],c],(d[1],1))
        var1[0,c] = np.sum(w1 ** 2)/total_var * 100
        var2[0,c] = np.sum(w2 ** 2)/total_var * 100
        var[0,c] = np.sum(w ** 2)/total_var * 100
        ratio[0,c] = var2[0,c]/var1[0,c]

    if BestModel:
        var_path = f'{res_path}/variances.xlsx' 
        df = pd.DataFrame({'components':range(1, W.shape[1]+1),
                        'View1': list(var1[0,:]), 'View2': list(var2[0,:]), 
                        'Both': list(var[0,:]),   'View2/View1': list(ratio[0,:])})
        writer = pd.ExcelWriter(var_path, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Sheet1')
        writer.save()    

    relvar1 = np.zeros((1, W.shape[1])) 
    relvar2 = np.zeros((1, W.shape[1]))
    relvar = np.zeros((1, W.shape[1]))
    for j in range(0, W.shape[1]): 
        relvar1[0,j] = var1[0,j]/np.sum(var1[0,:]) * 100 
        relvar2[0,j] = var2[0,j]/np.sum(var2[0,:]) * 100 
        relvar[0,j] = var[0,j]/np.sum(var[0,:]) * 100 

    if BestModel:
        relvar_path = f'{res_path}/relative_variances.xlsx' 
        df = pd.DataFrame({'components':range(1, W.shape[1]+1),
                        'View1': list(relvar1[0,:]),'View2': list(relvar2[0,:]), 
                        'Both': list(relvar[0,:])})
        writer1 = pd.ExcelWriter(relvar_path, engine='xlsxwriter')
        df.to_excel(writer1, sheet_name='Sheet1')
        writer1.save()

    #Relevant components
    comps = np.arange(W.shape[1])
    brain = comps[relvar1[0] > spvar]
    clinical = comps[relvar2[0] > spvar]
    rel_comps = np.union1d(brain,clinical)
    var_relcomps = np.sum(var[0,rel_comps])
    r_relcomps = ratio[0,rel_comps]        

    return np.sum(var), var_relcomps, r_relcomps, rel_comps                        

def plot_results(ninit, X, ylabels, res_path):

    #Output file
    beh_dim = X[1].shape[1] 
    MSE = np.zeros((1,ninit))
    MSE_trainmean = np.zeros((1,ninit))
    MSE_beh = np.zeros((ninit, beh_dim))
    MSE_beh_trmean = np.zeros((ninit, beh_dim))
    if 'missing' in res_path:
        MSEmissing = np.zeros((1,ninit))  
        Corrmissing = np.zeros((1,ninit))    
    LB = np.zeros((1,ninit))

    #Create a dictionary for the parameters
    file_ext = '.svg'
    thr_alpha = 1000
    spvar = 7.5
    
    ofile = open(f'{res_path}/output_{spvar}.txt','w')    
    for i in range(ninit):
        
        print('\nInitialisation: ', i+1, file=ofile)
        print('------------------------------------------------', file=ofile)
        filepath = f'{res_path}Results_run{i+1}.dictionary'
        #Load file
        with open(filepath, 'rb') as parameters:
            res = pickle.load(parameters)  

        #Computational time
        #print('Computational time (hours): ', round(res.time_elapsed/3600), file=ofile)
        print('Number of components: ', res.k, file=ofile)
        
        #Lower bound
        LB[0,i] = res.L[-1]
        print('Lower bound: ', LB[0,i], file=ofile)

        #-Predictions 
        #---------------------------------------------------------------------
        #Predict missing values
        X_train = [[] for _ in range(res.s)]
        X_test = [[] for _ in range(res.s)]  
        for j in range(res.s):
            X_train[j] = X[j][res.indTrain,:]
            X_test[j] = X[j][res.indTest,:]
        Beh_trainmean = np.mean(X_train[1], axis=0)               
        if 'missing' in filepath:
            if 'view1' in filepath:
                obs_view = np.array([0, 1])
                v_miss = 0
            elif 'view2' in filepath:
                obs_view = np.array([1, 0])
                v_miss = 1
            
            #predict missing values
            if 'rows' in filepath:
                mask_miss = res.missing_rows            
                X_train[v_miss][mask_miss,:] = 'NaN'
                missing_pred = GFAtools(X_train, res, obs_view).PredictMissing(missRows=True)
                miss_true = np.ndarray.flatten(res.miss_true)
            elif 'random' in filepath:
                mask_miss = res.missing_mask            
                X_train[v_miss][mask_miss] = 'NaN'
                missing_pred = GFAtools(X_train, res, obs_view).PredictMissing()
                miss_true = res.miss_true[mask_miss]   
            miss_pred = np.ndarray.flatten(missing_pred[v_miss][mask_miss])
            MSEmissing[0,i] = np.mean((miss_true - miss_pred) ** 2)
            Corrmissing[0,i] = np.corrcoef(miss_true,miss_pred)[0,1]
            print('MSE for missing data: ', MSEmissing[0,i], file=ofile)
            print('Corr. for missing data: ', Corrmissing[0,i], file=ofile)
    
        obs_view = np.array([1, 0])
        vpred = np.array(np.where(obs_view == 0))
        if 'spherical' in filepath:
            noise = 'spherical'
        else:
            noise = 'diagonal'    
        X_pred = GFAtools(X_test, res, obs_view).PredictView(noise)

        #-Metrics
        #----------------------------------------------------------------------------------
        MSE[0,i] = np.sqrt(np.mean((X_test[vpred[0,0]] - X_pred) ** 2))
        MSE_trainmean[0,i] = np.sqrt(np.mean((X_test[vpred[0,0]] - Beh_trainmean) ** 2))
        #MSE for each dimension - predict view 2 from view 1
        for j in range(0, beh_dim):
            MSE_beh[i,j] = np.mean((X_test[vpred[0,0]][:,j] - X_pred[:,j]) ** 2)/np.mean(X_test[vpred[0,0]][:,j] ** 2)
            MSE_beh_trmean[i,j] = np.mean((X_test[vpred[0,0]][:,j] - Beh_trainmean[j]) ** 2)/np.mean(X_test[vpred[0,0]][:,j] ** 2)

        #Weights and total variance
        W1 = res.means_w[0]
        W2 = res.means_w[1]
        W = np.concatenate((W1, W2), axis=0) 

        if hasattr(res, 'total_var') is False:           
            if 'spherical' in filepath:
                S1 = 1/res.E_tau[0] * np.ones((1, W1.shape[0]))[0]
                S2 = 1/res.E_tau[1] * np.ones((1, W2.shape[0]))[0]
                S = np.diag(np.concatenate((S1, S2), axis=0))
            else:
                S1 = 1/res.E_tau[0]
                S2 = 1/res.E_tau[1]
                S = np.diag(np.concatenate((S1, S2), axis=1)[0,:])
            total_var = np.trace(np.dot(W,W.T) + S) 
            res.total_var = total_var
            with open(filepath, 'wb') as parameters:
                pickle.dump(res, parameters) 

        Total_ExpVar, RelComps_var, RelComps_ratio, ind_lowK = compute_variances(W, res.d, res.total_var, spvar, res_path)
        print('Total explained variance: ', Total_ExpVar, file=ofile)
        print('Explained variance by relevant components: ', RelComps_var, file=ofile)
        print('Relevant components: ', ind_lowK, file=ofile)
        np.set_printoptions(precision=2)
        print('Ratio relevant components: ', RelComps_ratio, file=ofile)

        """ if len(ind_lowK) > 0:
            #Save brain weights
            brain_weights = {"wx": W1[:,ind_lowK]}
            io.savemat(f'{res_path}/wx{i+1}.mat', brain_weights)
            #Save clinical weights
            clinical_weights = {"wy": W2[:,ind_lowK]}
            io.savemat(f'{res_path}/wy{i+1}.mat', clinical_weights) """                 
    
    best_LB = int(np.argmax(LB)+1)
    print('\nOverall results for the best model---------------', file=ofile)
    print('-------------------------------------------------', file=ofile)   
    print('Best initialisation (Lower bound): ', best_LB, file=ofile)

    filepath = f'{res_path}Results_run{best_LB}.dictionary'
    with open(filepath, 'rb') as parameters:
        b_res = pickle.load(parameters)

    #Plot lower bound
    L_path = f'{res_path}/LB{file_ext}'
    plt.figure()
    plt.title('Lower Bound')
    plt.plot(res.L[1:])
    plt.savefig(L_path)
    plt.close()        

    #Weights and total variance
    W1 = b_res.means_w[0]
    W2 = b_res.means_w[1]
    W_best = np.concatenate((W1, W2), axis=0) 

    """ if hasattr(b_res, 'total_var') is False:           
        if 'spherical' in filepath:
            S1 = 1/b_res.E_tau[0] * np.ones((1, W1.shape[0]))[0]
            S2 = 1/b_res.E_tau[1] * np.ones((1, W2.shape[0]))[0]
            S = np.diag(np.concatenate((S1, S2), axis=0))
        else:
            S1 = 1/b_res.E_tau[0]
            S2 = 1/b_res.E_tau[1]
            S = np.diag(np.concatenate((S1, S2), axis=1)[0,:])
        total_var = np.trace(np.dot(W_best,W_best.T) + S) 
        b_res.total_var = total_var
        with open(filepath, 'wb') as parameters:
            pickle.dump(b_res, parameters) """ 
            
    #Compute variances
    ind_alpha1 = []
    ind_alpha2 = []  
    for k in range(W_best.shape[1]):
        if b_res.E_alpha[0][k] < thr_alpha:
            ind_alpha1.append(k)    
        if b_res.E_alpha[1][k] < thr_alpha:
            ind_alpha2.append(k)
       
    Total_ExpVar, RelComps_var, RelComps_ratio, ind_lowK = compute_variances(W_best, b_res.d, b_res.total_var, spvar, res_path, BestModel=True)
    print('Total explained variance: ', Total_ExpVar, file=ofile)
    print('Explained variance by relevant components: ', RelComps_var, file=ofile)
    print('Relevant components: ', ind_lowK, file=ofile)
    np.set_printoptions(precision=2)
    print('Ratio relevant components: ', RelComps_ratio, file=ofile)
 
    #np.set_printoptions(precision=2,suppress=True)
    #print('Alphas of rel. components (brain): ', np.round(b_res.E_alpha[0][ind_lowK], 1), file=ofile)
    #print('Alphas of rel. components (clinical): ', np.round(b_res.E_alpha[1][ind_lowK], 1), file=ofile)

    """ #plot relevant weights                     
    W_path = f'{res_path}/W_relevant{best_LB}_{spvar}{file_ext}'      
    plot_weights(W_best[:,ind_lowK], b_res.d, W_path)

    #plot relevant alphas
    a_path = f'{res_path}/alphas_relevant{best_LB}_{spvar}{file_ext}'
    a1 = np.reshape(b_res.E_alpha[0], (b_res.k, 1))
    a2 = np.reshape(b_res.E_alpha[1], (b_res.k, 1))
    a = np.concatenate((a1, a2), axis=1)
    hinton_diag(-a[ind_lowK,:].T, a_path) """

    #Specific components
    #CHANGE THIS - NEW CRITERIA! RATIO INSTEAD OF ALPHAS
    ind1 = np.intersect1d(ind_alpha1,ind_lowK)  
    ind2 = np.intersect1d(ind_alpha2,ind_lowK)                                        
    print('Brain components: ', ind1, file=ofile)
    print('Clinical components: ', ind2, file=ofile)

    if len(ind_lowK) > 0:
        #Save brain weights
        brain_weights = {"wx": W1[:,ind_lowK]}
        io.savemat(f'{res_path}/wx.mat', brain_weights)
        #Save clinical weights
        clinical_weights = {"wy": W2[:,ind_lowK]}
        io.savemat(f'{res_path}/wy.mat', clinical_weights)

    #alpha histograms
    plt.figure()
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    for i in range(b_res.s):
        axs[i].hist(b_res.E_alpha[i], bins=40)
        axs[i].axvline(x=thr_alpha,color='r')
        axs[i].set_xlabel('alphas')
    axs[0].set_title('Brain')
    axs[1].set_title('Clinical')
    alpha_path = f'{res_path}/alphas_dist{file_ext}'
    plt.savefig(alpha_path)
    plt.close()              

    print(f'\nPredictions--------------------------------', file=ofile)
    if 'missing' in filepath:
        print('Missing data: ', file=ofile)
        print(f'\nAvg. MSE (Std MSE): {np.mean(MSEmissing)} ({np.std(MSEmissing)}) ', file=ofile)
        print(f'Avg. Corr (missing data): ', np.mean(Corrmissing), file=ofile)
        print(f'Std Corr(missing data): ', np.std(Corrmissing), file=ofile)  

    #print(f'\nAvg. MSE: ', np.mean(MSE), file=ofile)
    #print(f'Std MSE: ', np.std(MSE), file=ofile)
    #print(f'\nAvg. MSE(mean train): ', np.mean(MSE_trainmean), file=ofile)
    #print(f'Std MSE(mean train): ', np.std(MSE_trainmean), file=ofile)

    sort_beh = np.argsort(np.mean(MSE_beh, axis=0))
    top_var = 10
    print(f'\nTop {top_var} predicted variables: \n', file=ofile)
    for l in range(top_var):
        print(ylabels[sort_beh[l]], file=ofile)

    #Predictions for behaviour
    #---------------------------------------------
    plt.figure(figsize=(10,6))
    pred_path = f'{res_path}/Predictions{file_ext}'
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

    """ #Plot sorted predictions
    sort_MSE = np.sort(np.mean(MSE_beh, axis=0))
    plt.figure()
    plt.plot(x, sort_MSE)
    plt.axvline(x=top_var,color='r')
    plt.title('Sorted predictions')
    plt.xlabel('Features of view 2')
    plt.ylabel('relative MSE')
    pred2_path = f'{res_path}/sort_pred{file_ext}'
    plt.savefig(pred2_path)
    plt.close()  """
     
    ofile.close()
    print('Visualisation concluded!')
    # return relevant components and best model
    return b_res, ind_lowK, spvar       
