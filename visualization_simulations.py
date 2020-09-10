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

def compute_variances(W, d, total_var, spvar, res_dir, BestRun = False):

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
        var_path = f'{res_dir}/variances.xlsx' 
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
        relvar_path = f'{res_dir}/relative_variances.xlsx' 
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
    plt.figure()
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

def plot_Z(Z, sort_comps=None, flip=None, path=None, match=False):
    x = np.linspace(0, Z.shape[0], Z.shape[0])
    ncomp = Z.shape[1]
    fig = plt.figure(figsize=(8, 6))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    for j in range(ncomp):
        ax = fig.add_subplot(ncomp, 1, j+1)    
        if match:
            ax.scatter(x, Z[:, sort_comps[j]] * flip[j], s=4)
        else:
            ax.scatter(x, Z[:, j], s=4)
        ax.set_xticks([])
        ax.set_yticks([])       
    plt.savefig(path)
    plt.close()           

def plot_params(model, res_dir, args, best_run, data, plot_trueparams=True):
    #Concatenate parameters across data sources    
    W_est = np.zeros((np.sum(model.d),model.k))
    alphas_est = np.zeros((model.k, args.num_sources))
    if plot_trueparams:
        W_true = np.zeros((np.sum(model.d),model.k))
        alphas_true = np.zeros((model.k, args.num_sources))
    d = 0
    for m in range(args.num_sources):
        Dm = model.d[m]
        if plot_trueparams:
            alphas_true[:,m] = data['alpha'][m]
            W_true[d:d+Dm,:] = data['W'][m]
        alphas_est[:,m] = model.E_alpha[m]
        W_est[d:d+Dm,:] = model.means_w[m]
        d += Dm  

    #LOADING MATRICES
    if plot_trueparams:
        #plot true Ws
        W_path = f'{res_dir}/[{best_run+1}]W_true.png'
        plot_weights(W_true, model.d, W_path) 
    #plot estimated Ws
    if model.k == data['true_K']:
        #match true and estimated components
        W_est, comp_e, flip = match_comps(W_est, W_true) 
    if args.impMedian:                          
        W_path = f'{res_dir}/[{best_run+1}]W_est_median.png'
    else:
        W_path = f'{res_dir}/[{best_run+1}]W_est.png'           
    plot_weights(W_est, model.d, W_path)
    #Save variances and relative variances of the best run
    """ S = np.diag(np.concatenate((1/model.tau[0], 1/model.tau[1]), axis=0))
    total_var = np.trace(np.dot(W, W.T) + S) 
    compute_variances(W, best_run.d, total_var, spvar, res_dir, BestRun=True) """
    
    #LATENT VARIABLES
    if plot_trueparams:    
        #plot true latent 
        Z_path = f'{res_dir}/[{best_run+1}]Z_true.png'    
        plot_Z(data['Z'], path=Z_path, match=False)
    #plot estimated latent variables
    if args.impMedian:                          
        Z_path = f'{res_dir}/[{best_run+1}]Z_est_median.png'
    else:
        Z_path = f'{res_dir}/[{best_run+1}]Z_est.png'
    if model.k == data['true_K']:
        plot_Z(model.means_z, comp_e, flip, Z_path, match=True)
    else:     
        plot_Z(model.means_z, path=Z_path)       

    #ALPHAS
    if plot_trueparams:
        #plot true alphas
        alphas_path = f'{res_dir}/[{best_run+1}]alphas_true.png'
        hinton_diag(-alphas_true.T, alphas_path)     
    #plot estimated alphas
    if args.impMedian:                          
        alphas_path = f'{res_dir}/[{best_run+1}]alphas_est_median.png'
    else:
        alphas_path = f'{res_dir}/[{best_run+1}]alphas_est.png'
    if model.k == data['true_K']:
        hinton_diag(-alphas_est[comp_e,:].T, alphas_path) 
    else:
        hinton_diag(-alphas_est.T, alphas_path)

    #plot ELBO
    L_path = f'{res_dir}/[{best_run+1}]ELBO.png'
    plt.figure(figsize=(5, 4))
    plt.plot(model.L[1:])
    plt.savefig(L_path)
    plt.close() 

def main_results(args, res_dir, InfoMiss=None):
       
    #Number of runs
    nruns = args.num_runs   
    #Initialise file where the results will be written
    ofile = open(f'{res_dir}/results.txt','w')    
    #Initialise variables to save MSEs, correlations and ELBO values
    MSE = np.zeros((1, nruns))
    MSE_chlv = np.zeros((1, nruns))
    if 'incomplete' in args.scenario:
        Corr_miss = np.zeros((len(InfoMiss['group']), nruns)) 
    if args.impMedian:
        MSEmed = np.zeros((1, nruns))          
    ELBO = np.zeros((1, nruns))    
    #Set thresholds to select relevant components 
    rel_var = 7.5 #relative variance
    r_var = 4 #ratio between group-specific variances explained  
    for i in range(0, nruns):
        print('\nInitialisation: ', i+1, file=ofile)

        #Load files
        model_file = f'{res_dir}/[{i+1}]ModelOutput.dictionary'
        with open(model_file, 'rb') as parameters:
            GFAotp = pickle.load(parameters)
        if args.impMedian: 
            model_median_file = f'{res_dir}/[{i+1}]ModelOutput_median.dictionary'
            with open(model_median_file, 'rb') as parameters:
                GFAotp_median = pickle.load(parameters)

        #Print ELBO 
        ELBO[0,i] = GFAotp.L[-1]
        print(f'Lower bound:', ELBO[0,i], file=ofile)
        #Print estimated taus
        for g in range(GFAotp.s):
            print(f'Estimated avg. taus (data source {g+1}):', np.around(np.mean(GFAotp.E_tau[g]),2), file=ofile)                             

        #Compute total variance explained
        T = np.zeros((1,np.sum(GFAotp.d)))
        W = np.zeros((np.sum(GFAotp.d),GFAotp.k))
        d = 0
        for m in range(args.num_sources):
            Dm = GFAotp.d[m]
            if 'diagonal' in args.noise:       
                T[0,d:d+Dm] = 1/GFAotp.E_tau[m]
            else:
                T[0,d:d+Dm] = np.ones((1,Dm)) * 1/GFAotp.E_tau[m]
            W[d:d+Dm,:] = GFAotp.means_w[m]
            d += Dm          
        T = np.diag(T[0,:])
        total_var = np.trace(np.dot(W,W.T) + T) 
        #Find relevant components
        """ Total_ExpVar, RelComps_var, RelComps_ratio, RelComps = compute_variances(W, GFAotp.d, total_var, spvar, res_dir)
        print('Total explained variance: ', Total_ExpVar, file=ofile)
        print('Explained variance by relevant components: ', RelComps_var, file=ofile)
        print('Relevant components: ', RelComps, file=ofile)
        np.set_printoptions(precision=2)
        print('Ratio relevant components: ', RelComps_ratio, file=ofile)   """      

        #Predictions
        #----------------------------------------------------------
        MSE[0,i] = GFAotp.MSE
        MSE_chlv[0,i] = GFAotp.MSE_chlev
        if args.impMedian:
            MSEmed[0,i] = GFAotp_median.MSE
        #Save correlation between true and predicted missing values
        if 'incomplete' in args.scenario:
            for j in range(len(InfoMiss['group'])):
                Corr_miss[j,i] = GFAotp.Corr_miss[0,j]                      

    #Plot results for best run
    #-------------------------------------------------------------------------------
    #Get best run
    best_run = int(np.argmax(ELBO))
    print('\nOVERALL RESULTS--------------------------', file=ofile)   
    print('Best run: ', best_run+1, file=ofile)
    #Load model output and data files of the best run
    model_file = f'{res_dir}/[{best_run+1}]ModelOutput.dictionary'
    with open(model_file, 'rb') as parameters:
        GFAotp_best = pickle.load(parameters)
    data_file = f'{res_dir}/[{best_run+1}]Data.dictionary'    
    with open(data_file, 'rb') as parameters:
        data = pickle.load(parameters) 

    #Plot true and estimated parameters
    plot_params(GFAotp_best, res_dir, args, best_run, data=data)                         

    #Multi-output predictions
    #--------------------------------------------------------------------------------
    print('\nPredictions-----------------',file=ofile)
    print('Model with observed data only:',file=ofile)
    print('Avg. MSE: ', np.around(np.mean(MSE, axis=1),2), file=ofile)
    print('Std MSE: ', np.around(np.std(MSE, axis=1),2), file=ofile)
    print('\nChance level:',file=ofile)
    print('Avg. MSE: ', np.around(np.mean(MSE_chlv,axis=1),2), file=ofile)
    print('Std MSE: ', np.around(np.std(MSE_chlv,axis=1),2), file=ofile)
    #Missing data prediction
    if 'incomplete' in args.scenario:
        print('\nPredictions for missing data -----------------',file=ofile)
        for j in range(len(InfoMiss['group'])):
            print('Data source: ',j+1, file=ofile)
            print('Avg. correlation: ', np.around(np.mean(Corr_miss[j,:]),2), file=ofile)
            print('Std. correlation: ', np.around(np.std(Corr_miss[j,:]),2), file=ofile)    

    if args.impMedian:
        print('\nModel with median imputation------------',file=ofile)
        #Load model output and data files of the best run
        model_file = f'{res_dir}/[{best_run+1}]ModelOutput_median.dictionary'
        with open(model_file, 'rb') as parameters:
            GFAotp_median_best = pickle.load(parameters)

        #taus
        for g in range(args.num_sources):
            np.set_printoptions(precision=2)
            print(f'Estimated avg. taus (data source {g+1}):', np.around(np.mean(GFAotp_median_best.E_tau[g]),2), file=ofile)

        #plot estimated parameters
        plot_params(GFAotp_median_best, res_dir, args, best_run, data, plot_trueparams=False)

        #predictions
        np.set_printoptions(precision=2)
        print('Predictions:',file=ofile)
        print('Avg. MSE: ', np.around(np.mean(MSEmed),2), file=ofile)
        print('Std MSE: ', np.around(np.std(MSEmed),2), file=ofile)                               
    ofile.close() 
        

        
   