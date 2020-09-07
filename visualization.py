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

def hinton_diag(matrix, path, max_weight=None, ax=None):
    # Draw Hinton diagram for visualizing a weight matrix.
    plt.figure() #figsize=(2, 1.5)
    ax = ax if ax is not None else plt.gca()
    fcolor = 'white'
    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor(fcolor)
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()
    plt.savefig(path)
    plt.close()

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

def match_comps(tempW,W_true):
    W = np.zeros((tempW.shape[0],tempW.shape[1]))
    cos = np.zeros((tempW.shape[1], W_true.shape[1]))
    for k in range(W_true.shape[1]):
        for j in range(tempW.shape[1]):
            cos[j,k] = cosine_similarity([W_true[:,k]],[tempW[:,j]])
    comp_e = np.argmax(np.absolute(cos),axis=0)
    flip = []       
    for comp in range(comp_e.size):
        if cos[comp_e[comp],comp] > 0:
            W[:,comp] = tempW[:,comp_e[comp]]
            flip.append(1)
        elif cos[comp_e[comp],comp] < 0:
            W[:,comp] =  - tempW[:,comp_e[comp]]
            flip.append(-1)
    return W, comp_e, flip                         

def plot_weights(W, d, W_path):
    x = np.linspace(1,3,num=W.shape[0])
    step = 6
    c = step * W.shape[1]+1
    plt.figure() #figsize=(2.5, 1)
    for col in range(W.shape[1]):
        y = W[:,col] + (col+c)
        c-=step
        # multiple line plot
        plt.plot( x, y, color='black', linewidth=1.5)
    fig = plt.gca()
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.axvline(x=x[d[0]-1],color='red')   
    plt.savefig(W_path)
    plt.close()

def plot_Z(model, sort_comps=None, flip=None, path=None, match=True):
    x = np.linspace(0, model.means_z.shape[0], model.means_z.shape[0])
    if 'est' in path:
        ncomp = model.means_z.shape[1]
    else:
        ncomp = model.Z.shape[1]
    fig = plt.figure(figsize=(8, 6))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    for j in range(ncomp):
        ax = fig.add_subplot(ncomp, 1, j+1)
        if 'true' in path:
            ax.scatter(x, model.Z[:, j], s=4)
        else:    
            if match and ncomp==model.k_true:
                ax.scatter(x, model.means_z[:, sort_comps[j]] * flip[j], s=4)
            else:
                ax.scatter(x, model.means_z[:, j], s=4)
        ax.set_xticks([])
        ax.set_yticks([])       
    plt.savefig(path)
    plt.close()    

def results_HCP(ninit, X, ylabels, res_path):

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

def results_simulations(ninit, res_path):
    
    #Load file
    filepath = f'{res_path}/Results_{ninit}runs.dictionary'
    with open(filepath, 'rb') as parameters:
        res = pickle.load(parameters)
    
    #Output file
    ofile = open(f'{res_path}/output.txt','w')    

    #Initialise values
    MSE_v1 = np.zeros((1, ninit))
    MSE_v2 = np.zeros((1, ninit))
    MSE_v1_tr = np.zeros((1, ninit))
    MSE_v2_tr = np.zeros((1, ninit))
    if 'missing' in filepath:
        MSEmed_v1 = np.zeros((1, ninit))
        MSEmed_v2 = np.zeros((1, ninit))
        Corr_miss = np.zeros((len(res[0].vmiss), ninit))       
    LB = np.zeros((1, ninit))    
    file_ext = '.svg'
    spvar = 7.5

    # Cycle for number of repetitions
    for i in range(0, ninit):
        print('\nInitialisation: ', i+1, file=ofile)   
        if 'missing' in filepath:
            for j in range(len(res[i].remove)):
                Corr_miss[j,i] = res[i].Corrmissing[j]

            file_median = f'{res_path}/Results_median.dictionary'
            with open(file_median, 'rb') as parameters:
                res1 = pickle.load(parameters)

        #LB values
        LB[0,i] = res[i].L[-1]
        print(f'Lower bound:', LB[0,i], file=ofile)        

        #Predictions
        #---------------------------------------------
        MSE_v1[0,i] = res[i].MSE1
        MSE_v2[0,i] = res[i].MSE2
        MSE_v1_tr[0,i] = res[i].MSE1_train
        MSE_v2_tr[0,i] = res[i].MSE2_train
        if 'missing' in filepath:
            MSEmed_v1[0,i] = res1[i].MSE1
            MSEmed_v2[0,i] = res1[i].MSE2                   

        #taus
        for v in range(res[i].s):
            print(f'Estimated avg. taus (view {v+1}):', np.mean(res[i].E_tau[v]), file=ofile)                             

        #plot estimated Ws
        W_true = np.concatenate((res[i].W[0], res[i].W[1]), axis=0)
        W = np.concatenate((res[i].means_w[0], res[i].means_w[1]), axis=0)
        if W.shape[1] == W_true.shape[1]:
            W, comp_e, flip = match_comps(W, W_true)                       
        W_path = f'{res_path}/W_est{i+1}.png'      
        plot_weights(W, res[i].d, W_path)

        #Compute true variances       
        S1 = 1/res[i].tau[0]
        S2 = 1/res[i].tau[1]
        S = np.diag(np.concatenate((S1, S2), axis=0))
        total_var = np.trace(np.dot(W_true,W_true.T) + S) 
        Total_ExpVar, RelComps_var, RelComps_ratio, ind_lowK = compute_variances(W, res[i].d, total_var, spvar, res_path)
        print('Total explained variance: ', Total_ExpVar, file=ofile)
        print('Explained variance by relevant components: ', RelComps_var, file=ofile)
        print('Relevant components: ', ind_lowK, file=ofile)
        np.set_printoptions(precision=2)
        print('Ratio relevant components: ', RelComps_ratio, file=ofile)


    #Overall results
    best_init = int(np.argmax(LB))
    print('\nOverall results--------------------------', file=ofile)   
    print('Best initialisation: ', best_init+1, file=ofile)

    # plot true Ws
    W_true = np.concatenate((res[best_init].W[0], res[best_init].W[1]), axis=0)
    W_path = f'{res_path}/W_true{best_init+1}{file_ext}'
    plot_weights(W_true, res[best_init].d, W_path) 

    #plot estimated Ws
    W = np.concatenate((res[best_init].means_w[0], res[best_init].means_w[1]), axis=0)
    if W.shape[1] == W_true.shape[1]:
        W, comp_e, flip = match_comps(W, W_true)                       
    W_path = f'{res_path}/W_est{best_init+1}{file_ext}'      
    plot_weights(W, res[best_init].d, W_path)

    #Save variances and relative variances
    S1 = 1/res[best_init].tau[0]
    S2 = 1/res[best_init].tau[1]
    S = np.diag(np.concatenate((S1, S2), axis=0))
    total_var = np.trace(np.dot(W_true, W_true.T) + S) 
    compute_variances(W, res[best_init].d, total_var, spvar, res_path, BestModel=True)

    # plot true latent variables
    Z_path = f'{res_path}/Z_true{best_init+1}{file_ext}'
    if W.shape[1] == W_true.shape[1]:
        plot_Z(res[best_init], comp_e, flip, Z_path)
    else:     
        plot_Z(res[best_init], path=Z_path, match=False)

    # plot estimated latent variables
    Z_path = f'{res_path}/Z_est{best_init+1}{file_ext}'
    if W.shape[1] == W_true.shape[1]:
        plot_Z(res[best_init], comp_e, flip, Z_path)
    else:     
        plot_Z(res[best_init], path=Z_path, match=False)       

    #plot true alphas
    a_path = f'{res_path}/alphas_true{best_init+1}{file_ext}'
    a1 = np.reshape(res[best_init].alphas[0], (res[best_init].alphas[0].shape[0], 1))
    a2 = np.reshape(res[best_init].alphas[1], (res[best_init].alphas[1].shape[0], 1))
    a = np.concatenate((a1, a2), axis=1)
    hinton_diag(-a.T, a_path)     

    #plot estimated alphas
    a_path = f'{res_path}/alphas_est{best_init+1}{file_ext}'
    a1 = np.reshape(res[best_init].E_alpha[0], (res[best_init].k, 1))
    a2 = np.reshape(res[best_init].E_alpha[1], (res[best_init].k, 1))
    a = np.concatenate((a1, a2), axis=1)
    if W.shape[1] == W_true.shape[1]:
        hinton_diag(-a[comp_e,:].T, a_path) 
    else:
        hinton_diag(-a.T, a_path)

    # plot lower bound
    L_path = f'{res_path}/LB{best_init+1}{file_ext}'
    plt.figure(figsize=(5, 4))
    plt.plot(res[best_init].L[1:])
    plt.savefig(L_path)
    plt.close()                      

    #Predictions
    #View 1
    print('\nPredictions for view 1-----------------',file=ofile)
    print('Observed data:',file=ofile)
    print(f'Avg. MSE: ', np.mean(MSE_v1), file=ofile)
    print(f'Std MSE: ', np.std(MSE_v1), file=ofile)
    print('\nTrain data:',file=ofile)
    print(f'Avg. MSE: ', np.mean(MSE_v1_tr), file=ofile)
    print(f'Std MSE: ', np.std(MSE_v1_tr), file=ofile) 
    if 'missing' in filepath: 
        print('\nImputed data with median:',file=ofile)
        print(f'Avg. MSE: ', np.mean(MSEmed_v1), file=ofile)
        print(f'Std MSE: ', np.std(MSEmed_v1), file=ofile)

    #View 2
    print('\nPredictions for view 2-----------------',file=ofile)
    print('Observed data:',file=ofile)
    print(f'Avg. MSE: ', np.mean(MSE_v2), file=ofile)
    print(f'Std MSE: ', np.std(MSE_v2), file=ofile)
    print('\nTrain data:',file=ofile)
    print(f'Avg. MSE: ', np.mean(MSE_v2_tr), file=ofile)
    print(f'Std MSE: ', np.std(MSE_v2_tr), file=ofile)
    if 'missing' in filepath:   
        print('\nImputed data with median:',file=ofile)
        print(f'Avg. MSE: ', np.mean(MSEmed_v2), file=ofile)
        print(f'Std MSE: ', np.std(MSEmed_v2), file=ofile)          

    if 'missing' in filepath:
        print('\nPredictions for missing data -----------------',file=ofile)
        for i in range(len(res[0].vmiss)):
            print(f'Avg. MSE (missing data in view {res[0].vmiss[i]}): ', np.mean(MSE_miss[i,:]), file=ofile)
            print(f'Std MSE (missing data in view {res[0].vmiss[i]}): ', np.std(MSE_miss[i,:]), file=ofile)
            print(f'Avg. Corr (missing data): ', np.mean(Corr_miss[i,:]), file=ofile)
            print(f'Std Corr(missing data): ', np.std(Corr_miss[i,:]), file=ofile) 

        W_true = np.concatenate((res[best_init].W[0], res[best_init].W[1]), axis=0)   
        
        #MODEL MEDIAN
        #--------------------------------------------------------------------------------------
        #taus
        for v in range(res1[best_init].s):
            print(f'Estimated avg. taus (view {v+1}):', np.mean(res1[best_init].E_tau[v]), file=ofile) 

        #plot estimated projections
        W = np.concatenate((res1[best_init].means_w[0], res1[best_init].means_w[1]), axis=0)
        if W.shape[1] == W_true.shape[1]:
            W, comp_e, flip = match_comps(W, W_true)                     
        W_path = f'{res_path}/W_est_MEDIAN{file_ext}'      
        plot_weights(W, res[0].d, W_path)

        # plot estimated latent variables
        Z_path = f'{res_path}/Z_est_MEDIAN{file_ext}'
        if W.shape[1] == W_true.shape[1]:
            plot_Z(res1[best_init], comp_e, flip, Z_path)
        else:     
            plot_Z(res1[best_init], path=Z_path, match=False) 

        #plot estimated alphas
        a_path = f'{res_path}/alphas_est_MEDIAN{file_ext}'
        a1 = np.reshape(res1[best_init].E_alpha[0], (res1[best_init].k, 1))
        a2 = np.reshape(res1[best_init].E_alpha[1], (res1[best_init].k, 1))
        a = np.concatenate((a1, a2), axis=1)
        if W.shape[1] == W_true.shape[1]:
            hinton_diag(-a[comp_e,:].T, a_path) 
        else:
            hinton_diag(-a.T, a_path)

        # plot lower bound
        L_path = f'{res_path}/LB_MEDIAN{file_ext}'
        plt.figure(figsize=(5, 4))
        plt.plot(res1[best_init].L[1:])
        plt.savefig(L_path)
        plt.close()                            

    ofile.close() 
        

        
   