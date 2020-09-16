import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from scipy import io
from utils import GFAtools
from sklearn.metrics.pairwise import cosine_similarity

def find_relfactors(W, model, total_var, thrs, res_dir):
    #Calculate explained variance for each factor across 
    # and within data sources 
    ncomps = W.shape[1]
    var_within = np.zeros((model.s, ncomps))
    d=0
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
    for c in range(ncomps):
        ratio = var_within[1,c]/var_within[0,c]
        if np.any(relvar_within[:,c] > thrs['rel_var']):
            if ratio > 400:
                relfactors_specific[1].append(c+1)
            elif ratio < 0.001:
                relfactors_specific[0].append(c+1)
            else:
                relfactors_shared.append(c+1)               

    return relfactors_shared, relfactors_specific

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

def main_results(args, X, ylabels, res_path):

    nruns = args.num_runs #number of runs
    #initialise variables to save MSEs, correlations and ELBO values
    MSE_beh = np.zeros((nruns, X[1].shape[1]))
    MSE_beh_trmean = np.zeros((nruns, X[1].shape[1]))
    if args.scenario == 'incomplete': 
        Corr_miss = np.zeros((1,nruns))    
    ELBO = np.zeros((1,nruns))
    #Set thresholds to select relevant components 
    thrs = {'rel_var': 7.5, 'r_var': 4} #relative variance and ratio between group-specific variances 
    #initialise file where the results will be written
    ofile = open(f'{res_path}/results.txt','w')   
    
    for i in range(nruns):  
        print('\nInitialisation: ', i+1, file=ofile)
        print('------------------------------------------------', file=ofile)
        filepath = f'{res_path}[{i+1}]Results.dictionary'
        #Load file
        with open(filepath, 'rb') as parameters:
            GFA_otp = pickle.load(parameters)  

        #Print computational time and total number of components
        print('Computational time (minutes): ', round(GFA_otp.time_elapsed/60), file=ofile)
        print('Number of components: ', GFA_otp.k, file=ofile)
        #Lower bound
        ELBO[0,i] = GFA_otp.L[-1]
        print('ELBO (last value):', np.around(ELBO[0,i],2), file=ofile)

        #-Predictions (predict group 2 from group 1)
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
        # Get means of the behavioural/demographic variables (data source 2)
        Beh_trainmean = np.nanmean(X_train[1], axis=0)               
        # MSE for each behavioural/demographic variable
        obs_group = np.array([1, 0]) #group 1 was observed 
        gpred = np.where(obs_group == 0)[0][0] #get the non-observed group  
        X_pred = GFAtools(X_test, GFA_otp).PredictView(obs_group, args.noise)
        for j in range(gpred.size):
            MSE_beh[i,j] = np.mean((X_test[gpred][:,j] - X_pred[0][:,j]) ** 2)/np.mean(X_test[gpred][:,j] ** 2)
            MSE_beh_trmean[i,j] = np.mean((X_test[gpred][:,j] - Beh_trainmean[j]) ** 2)/np.mean(X_test[gpred][:,j] ** 2)
        
        # Predict missing values
        if args.scenario == 'incomplete':
            infmiss = {'perc': [args.pmiss], #percentage of missing data 
                'type': [args.tmiss], #type of missing data 
                'group': [args.gmiss]} #groups that will have missing values          
            miss_pred = GFAtools(X_train, GFA_otp).PredictMissing(args.num_sources, infmiss)
            miss_true = GFA_otp.miss_true
            Corr_miss[0,i] = np.corrcoef(miss_true[miss_true != 0], miss_pred[0][miss_pred[0] != 0])[0,1]

        #Weights and total variance
        W = np.concatenate((GFA_otp.means_w[0], GFA_otp.means_w[1]), axis=0) 
        if hasattr(GFA_otp, 'total_var') is False:           
            if 'spherical' in filepath:
                S1 = 1/GFA_otp.E_tau[0] * np.ones((1, W1.shape[0]))[0]
                S2 = 1/GFA_otp.E_tau[1] * np.ones((1, W2.shape[0]))[0]
                S = np.diag(np.concatenate((S1, S2), axis=0))
            else:
                S1 = 1/GFA_otp.E_tau[0]
                S2 = 1/GFA_otp.E_tau[1]
                S = np.diag(np.concatenate((S1, S2), axis=1)[0,:])
            total_var = np.trace(np.dot(W,W.T) + S) 
            GFA_otp.total_var = total_var
            with open(filepath, 'wb') as parameters:
                pickle.dump(GFA_otp, parameters) 

        Total_ExpVar, RelComps_var, RelComps_ratio, ind_lowK = compute_variances(W, GFA_otp.d, GFA_otp.total_var, spvar, GFA_otp_path)
        print('Total explained variance: ', Total_ExpVar, file=ofile)
        print('Explained variance by relevant components: ', RelComps_var, file=ofile)
        print('Relevant components: ', ind_lowK, file=ofile)
        np.set_printoptions(precision=2)
        print('Ratio relevant components: ', RelComps_ratio, file=ofile)

        """ var_relcomps = np.sum(expvar_allcomps(np.array(relcomps_sh)))
        for m in range(args.num_sources):
            var_relcomps += np.sum(expvar_allcomps(np.array(relcomps_sp[m])))
        print('Variance explained by relevant components: ', var_relcomps, file=ofile)  """                 
    
    best_ELBO = int(np.argmax(ELBO)+1)
    print('\nOverall results for the best model---------------', file=ofile)
    print('-------------------------------------------------', file=ofile)   
    print('Best initialisation (Lower bound): ', best_ELBO, file=ofile)

    filepath = f'{res_path}Results_run{best_LB}.dictionary'
    with open(filepath, 'rb') as parameters:
        GFA_botp = pickle.load(parameters)

    #Plot lower bound
    L_path = f'{res_path}/LB{file_ext}'
    plt.figure()
    plt.title('Lower Bound')
    plt.plot(res.L[1:])
    plt.savefig(L_path)
    plt.close()        

    #Weights and total variance
    W1 = GFA_botp.means_w[0]
    W2 = GFA_botp.means_w[1]
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
       
    Total_ExpVar, RelComps_var, RelComps_ratio, ind_lowK = compute_variances(W_best, GFA_botp.d, GFA_botp.total_var, spvar, res_path, BestModel=True)
    print('Total explained variance: ', Total_ExpVar, file=ofile)
    print('Explained variance by relevant components: ', RelComps_var, file=ofile)
    print('Relevant components: ', ind_lowK, file=ofile)
    np.set_printoptions(precision=2)
    print('Ratio relevant components: ', RelComps_ratio, file=ofile)

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
    for i in range(GFA_botp.s):
        axs[i].hist(GFA_botp.E_alpha[i], bins=40)
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
     
    ofile.close()
    print('Visualisation concluded!')       
