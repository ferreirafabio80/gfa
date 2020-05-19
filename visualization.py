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

def plot_predictions(df, ymax, title,path):
    # style
    plt.style.use('seaborn-darkgrid')
    
    # create a color palette
    palette = plt.get_cmap('Set1')
    
    # multiple line plot
    num=0
    for column in df.drop('x', axis=1):
        num+=1
        plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)
    
    # Add legend
    plt.legend(loc=2, ncol=2)
    
    # Add titles
    plt.title(title, loc='center', fontsize=14, fontweight=0)
    plt.xlabel("Dimensions of W")
    plt.ylabel("Relative MMSE")
    plt.ylim([0,ymax+0.3])
    plt.savefig(path)
    plt.close()

def compute_variances(W, d, total_var, spvar, res_path, BestModel=False):

    #Explained variance
    var1 = np.zeros((1, W.shape[1])) 
    var2 = np.zeros((1, W.shape[1])) 
    var = np.zeros((1, W.shape[1]))
    for c in range(0, W.shape[1]):
        w = np.reshape(W[:,c],(W.shape[0],1))
        w1 = np.reshape(W[0:d[0],c],(d[0],1))
        w2 = np.reshape(W[d[0]:d[0]+d[1],c],(d[1],1))
        var1[0,c] = (np.trace(np.dot(w1.T, w1))/total_var) * 100
        var2[0,c] = (np.trace(np.dot(w2.T, w2))/total_var) * 100
        var[0,c] = (np.trace(np.dot(w.T, w))/total_var) * 100

    if BestModel:
        relvar_path = f'{res_path}/variances.xlsx' 
        df = pd.DataFrame({'components':range(1, W.shape[1]+1),'Brain': list(var1[0,:]),'Behaviour': list(var2[0,:]), 'Both': list(var[0,:])})
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer1 = pd.ExcelWriter(relvar_path, engine='xlsxwriter')
        # Convert the dataframe to an XlsxWriter Excel object.
        df.to_excel(writer1, sheet_name='Sheet1')
        # Close the Pandas Excel writer and output the Excel file.
        writer1.save()    

    relvar1 = np.zeros((1, W.shape[1])) 
    relvar2 = np.zeros((1, W.shape[1]))
    relvar = np.zeros((1, W.shape[1]))
    for j in range(0, W.shape[1]):
        relvar1[0,j] = 100 - ((np.sum(var1[0,:]) - var1[0,j])/np.sum(var1[0,:])) * 100 
        relvar2[0,j] = 100 - ((np.sum(var2[0,:]) - var2[0,j])/np.sum(var2[0,:])) * 100  
        relvar[0,j] = 100 - ((np.sum(var[0,:]) - var[0,j])/np.sum(var[0,:])) * 100  

    if BestModel:
        relvar_path = f'{res_path}/relative_variances.xlsx' 
        df = pd.DataFrame({'components':range(1, W.shape[1]+1),'Brain': list(relvar1[0,:]),'Behaviour': list(relvar2[0,:]), 'Both': list(relvar[0,:])})
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer1 = pd.ExcelWriter(relvar_path, engine='xlsxwriter')
        # Convert the dataframe to an XlsxWriter Excel object.
        df.to_excel(writer1, sheet_name='Sheet1')
        # Close the Pandas Excel writer and output the Excel file.
        writer1.save()

        #Plot relative variance per component
        relvar_fig = f'{res_path}/relvar.png'  
        x = np.arange(W.shape[1])    
        plt.figure(figsize=(20,10))
        fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)
        axs[0].plot(x, relvar1[0])
        axs[0].axhline(y=spvar,color='r')
        axs[0].set_title('Brain rel. variance')
        axs[1].plot(x, relvar2[0])
        axs[1].axhline(y=spvar,color='r')
        axs[1].set_title('Behaviour rel. variance')  
        plt.savefig(relvar_fig)
        plt.close()

    #Relevant components
    comps = np.arange(W.shape[1])
    brain = comps[relvar1[0] > spvar]
    clinical = comps[relvar2[0] > spvar]
    rel_comps = np.union1d(brain,clinical)
    var_relcomps = np.sum(var[0,rel_comps])        

    return np.sum(var), var_relcomps, rel_comps 

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
    fig = plt.figure(figsize=(4, 4))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    for j in range(ncomp):
        ax = fig.add_subplot(ncomp, 1, j+1)
        if 'true' in path:
            ax.scatter(x, model.Z[:, j])
        else:    
            if match and ncomp==model.k_true:
                ax.scatter(x, model.means_z[:, sort_comps[j]] * flip[j])
            else:
                ax.scatter(x, model.means_z[:, j])
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
    LB = np.zeros((1,ninit))

    #Create a dictionary for the parameters
    file_ext = '.png'
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
        print('Computational time (hours): ', round(res.time_elapsed/3600), file=ofile)
        
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
            print('MSE for missing data: ', MSEmissing[0,i], file=ofile)
    
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

        Total_ExpVar, RelComps_var, ind_lowK = compute_variances(W, res.d, res.total_var, spvar, res_path)
        print('Total explained variance: ', Total_ExpVar, file=ofile)
        print('Explained variance by relevant components: ', RelComps_var, file=ofile)
        print('Relevant components: ', ind_lowK, file=ofile)

        if len(ind_lowK) > 0:
            #Save brain weights
            brain_weights = {"wx": W1[:,ind_lowK]}
            io.savemat(f'{res_path}/wx{i+1}.mat', brain_weights)
            #Save clinical weights
            clinical_weights = {"wy": W2[:,ind_lowK]}
            io.savemat(f'{res_path}/wy{i+1}.mat', clinical_weights)                 
    
    best_LB = int(np.argmax(LB)+1)
    print('\nOverall results for the best model--------------------------', file=ofile)   
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
            pickle.dump(b_res, parameters)  """
            
    #Compute variances
    ind_alpha1 = []
    ind_alpha2 = []  
    for k in range(W_best.shape[1]):
        if b_res.E_alpha[0][k] < thr_alpha:
            ind_alpha1.append(k)    
        if b_res.E_alpha[1][k] < thr_alpha:
            ind_alpha2.append(k)
       
    Total_ExpVar, RelComps_var, ind_lowK = compute_variances(W_best, b_res.d, b_res.total_var, spvar, res_path, BestModel=True)
    print('Total explained variance: ', Total_ExpVar, file=ofile)
    print('Explained variance by relevant components: ', RelComps_var, file=ofile)
    np.set_printoptions(precision=2,suppress=True)
    print('Alphas of rel. components (brain): ', np.round(b_res.E_alpha[0][ind_lowK], 1), file=ofile)
    print('Alphas of rel. components (clinical): ', np.round(b_res.E_alpha[1][ind_lowK], 1), file=ofile)

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
    ind1 = np.intersect1d(ind_alpha1,ind_lowK)  
    ind2 = np.intersect1d(ind_alpha2,ind_lowK)                                        
    print('Brain components: ', ind1, file=ofile)
    print('Clinical components: ', ind2, file=ofile)

    """ if len(ind_lowK) > 0:
        #Save brain weights
        brain_weights = {"wx": W1[:,ind_lowK]}
        io.savemat(f'{res_path}/wx.mat', brain_weights)
        #Save clinical weights
        clinical_weights = {"wy": W2[:,ind_lowK]}
        io.savemat(f'{res_path}/wy.mat', clinical_weights) """

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
        print(f'\nAvg. MSE (missing data): ', np.mean(MSEmissing), file=ofile)
        print(f'Std MSE(missing data): ', np.std(MSEmissing), file=ofile)  

    print(f'\nAvg. MSE: ', np.mean(MSE), file=ofile)
    print(f'Std MSE: ', np.std(MSE), file=ofile)
    print(f'\nAvg. MSE(mean train): ', np.mean(MSE_trainmean), file=ofile)
    print(f'Std MSE(mean train): ', np.std(MSE_trainmean), file=ofile)

    sort_beh = np.argsort(np.mean(MSE_beh, axis=0))
    top_var = 10
    print('\n---------------------------------', file=ofile)
    print(f'Top {top_var} predicted variables: \n', file=ofile)
    for l in range(top_var):
        print(ylabels[sort_beh[l]], file=ofile)

    #Predictions for behaviour
    #---------------------------------------------
    plt.figure(figsize=(10,8))
    pred_path = f'{res_path}/Predictions{file_ext}'
    x = np.arange(MSE_beh.shape[1])
    plt.errorbar(x, np.mean(MSE_beh,axis=0), yerr=np.std(MSE_beh,axis=0), fmt='bo', label='Predictions')
    plt.errorbar(x, np.mean(MSE_beh_trmean,axis=0), yerr=np.std(MSE_beh_trmean,axis=0), fmt='yo', label='Train mean')
    plt.legend(loc='upper right',fontsize=14)
    plt.ylim((0,2.5))
    plt.title('Full Model',fontsize=18)
    plt.xlabel('Features of view 2',fontsize=16)
    plt.ylabel('relative MSE',fontsize=16)
    plt.savefig(pred_path)
    plt.close()

    #Plot sorted predictions
    sort_MSE = np.sort(np.mean(MSE_beh, axis=0))
    plt.figure()
    plt.plot(x, sort_MSE)
    plt.axvline(x=top_var,color='r')
    plt.title('Sorted predictions')
    plt.xlabel('Features of view 2')
    plt.ylabel('relative MSE')
    pred2_path = f'{res_path}/sort_pred{file_ext}'
    plt.savefig(pred2_path)
    plt.close() 
     
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
    
    if 'missing' in filepath:
        MSE_miss = np.zeros((len(res[0].vmiss), ninit))

    MSE_v1 = np.zeros((1, ninit))
    MSE_v2 = np.zeros((1, ninit))
    if 'missing' in filepath:
        MSEimp_v1 = np.zeros((1, ninit))
        MSEimp_v2 = np.zeros((1, ninit))
        MSEmed_v1 = np.zeros((1, ninit))
        MSEmed_v2 = np.zeros((1, ninit))
        LB_imp = np.zeros((1, ninit)) 
        LB_med = np.zeros((1, ninit))         

    LB = np.zeros((1, ninit))    
    file_ext = '.png'
    for i in range(0, ninit):

        print('\nInitialisation: ', i+1, file=ofile)   
        if 'missing' in filepath:
            for j in range(len(res[i].remove)):
                print(f'MSE missing data (view {res[i].vmiss[j]}):', res[i].MSEmissing[j], file=ofile)
                MSE_miss[j,i] = res[i].MSEmissing[j]

            file_missing = f'{res_path}/Results_imputation.dictionary'
            with open(file_missing, 'rb') as parameters:
                res1 = pickle.load(parameters)

            file_median = f'{res_path}/Results_median.dictionary'
            with open(file_median, 'rb') as parameters:
                res2 = pickle.load(parameters)

            #Lower bound values for median and imputation models
            LB_imp[0,i] = res1[i].L[-1]                
            LB_med[0,i] = res2[i].L[-1]

        #Predictions for view 1
        #---------------------------------------------
        MSE_v1[0,i] = res[i].MSE1
        if 'missing' in filepath:
            MSEimp_v1[0,i] = res1[i].MSE1
            MSEmed_v1[0,i] = res2[i].MSE1

        #Predictions for view 2
        #---------------------------------------------
        MSE_v2[0,i] = res[i].MSE2
        if 'missing' in filepath:
            MSEimp_v2[0,i] = res1[i].MSE2
            MSEmed_v2[0,i] = res2[i].MSE2       
    
        #LB values
        LB[0,i] = res[i].L[-1]
        print(f'Lower bound:', LB[0,i], file=ofile)

        #taus
        for v in range(res[i].s):
            print(f'Estimated avg. taus (view {v+1}):', np.mean(res[i].E_tau[v]), file=ofile)                             

        # plot lower bound
        L_path = f'{res_path}/LB{i+1}{file_ext}'
        plt.figure(figsize=(5, 4))
        plt.plot(res[i].L[1:])
        plt.savefig(L_path)
        plt.close()

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

    #Predictions
    #View 1
    print('\nPredictions for view 1-----------------',file=ofile)
    print('Observed data:',file=ofile)
    print(f'Avg. MSE: ', np.mean(MSE_v1), file=ofile)
    print(f'Std MSE: ', np.std(MSE_v1), file=ofile) 
    if 'missing' in filepath: 
        print('\nImputed data with predicted values:',file=ofile)
        print(f'Avg. MSE: ', np.mean(MSEimp_v1), file=ofile)
        print(f'Std MSE: ', np.std(MSEimp_v1), file=ofile) 
        print('\nImputed data with median:',file=ofile)
        print(f'Avg. MSE: ', np.mean(MSEmed_v1), file=ofile)
        print(f'Std MSE: ', np.std(MSEmed_v1), file=ofile)

    #View 2
    print('\nPredictions for view 2-----------------',file=ofile)
    print('Observed data:',file=ofile)
    print(f'Avg. MSE: ', np.mean(MSE_v2), file=ofile)
    print(f'Std MSE: ', np.std(MSE_v2), file=ofile)
    if 'missing' in filepath:   
        print('\nImputed data with predicted values:',file=ofile)
        print(f'Avg. MSE: ', np.mean(MSEimp_v2), file=ofile)
        print(f'Std MSE: ', np.std(MSEimp_v2), file=ofile) 
        print('\nImputed data with median:',file=ofile)
        print(f'Avg. MSE: ', np.mean(MSEmed_v2), file=ofile)
        print(f'Std MSE: ', np.std(MSEmed_v2), file=ofile)          

    if 'missing' in filepath:
        for i in range(len(res[0].vmiss)):
            print(f'Avg. MSE (missing data in view {res[0].vmiss[i]}): ', np.mean(MSE_miss[i,:]), file=ofile)
            print(f'Std MSE (missing data in view {res[0].vmiss[i]}): ', np.std(MSE_miss[i,:]), file=ofile)

        W_true = np.concatenate((res[best_init].W[0], res[best_init].W[1]), axis=0)
        #MODEL IMPUTATION
        #---------------------------------------------------------------------------------------
        #plot estimated projections            
        W = np.concatenate((res1[best_init].means_w[0], res1[best_init].means_w[1]), axis=0)
        if W.shape[1] == W_true.shape[1]:
            W, comp_e, flip = match_comps(W, W_true)                   
        W_path = f'{res_path}/W_est_IMPUTATION{file_ext}'      
        plot_weights(W, res[0].d, W_path)

        # plot estimated latent variables
        Z_path = f'{res_path}/Z_est_IMPUTATION{file_ext}'
        if W.shape[1] == W_true.shape[1]:
            plot_Z(res1[best_init], comp_e, flip, Z_path)
        else:     
            plot_Z(res1[best_init], path=Z_path, match=False)   

        #MODEL MEDIAN
        #--------------------------------------------------------------------------------------
        #plot estimated projections
        W = np.concatenate((res2[best_init].means_w[0], res2[best_init].means_w[1]), axis=0)
        if W.shape[1] == W_true.shape[1]:
            W, comp_e, flip = match_comps(W, W_true)                     
        W_path = f'{res_path}/W_est_MEDIAN{file_ext}'      
        plot_weights(W, res[0].d, W_path)

        # plot estimated latent variables
        Z_path = f'{res_path}/Z_est_MEDIAN{file_ext}'
        if W.shape[1] == W_true.shape[1]:
            plot_Z(res2[best_init], comp_e, flip, Z_path)
        else:     
            plot_Z(res2[best_init], path=Z_path, match=False)                     

    ofile.close() 
    return res[best_init]   
        

        
   