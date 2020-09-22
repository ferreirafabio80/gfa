""" Module to save and plot results of the experiments on 
    synthetic data """

#Author: Fabio S. Ferreira (fabio.ferreira.16@ucl.ac.uk)
#Date: 17 September 2020

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def hinton_diag(matrix, path):

    """ 
    Draw Hinton diagram for visualizing a weight matrix.

    Parameters
    ----------
    matrix : array-like 
        Weight matrix.
    
    path : str
        Path to save the diagram. 
    
    """
    plt.figure()
    ax = plt.gca()
    fcolor = 'white'
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

def find_relfactors(W, model, total_var):
    
    """ 
    Find the most relevant factors.

    Parameters
    ----------
    W : array-like, shape(n_features, n_comps)
        Concatenated loading matrix. The number of features
        here correspond to the total number of features in 
        all groups. 

    model : Outputs of the model.

    total_var : float
        Total variance explained.

    Returns
    -------
    relfactors_shared : list
        A list of the relevant shared factors.

    relfactors_specific : list
        A list of the relevant factors specific to each group.
    
    """
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
        if np.any(relvar_within[:,c] > 7.5):
            if ratio > 400:
                relfactors_specific[1].append(c)
            elif ratio < 0.001:
                relfactors_specific[0].append(c)
            else:
                relfactors_shared.append(c)               

    return relfactors_shared, relfactors_specific

def match_factors(tempW, W_true):
    
    """ 
    Match the estimated factors to the true generated ones.

    Parameters
    ----------
    tempW : array-like, shape(n_features, n_comps)
        Concatenated estimated loading matrix. The number 
        of features here correspond to the total number of 
        features in all groups.

    W_true : array-like, shape(n_features, n_comps)
        Concatenated true loading matrix. The number of 
        features here correspond to the total number of 
        features in all groups.     

    Returns
    -------
    W : array-like, shape(n_features, n_comps)
        Sorted version of the estimated loading matrix.

    sim_factors : array-like, shape(n_comps,)
        Matching indices. These are obtained by calculating
        the maximum cosine similarity between estimated and
        true factors.

    flip : list
        Flip sign info. Positive cosine similarity corresponds
        to the same sign and negative cosine similarity 
        represents inverse sign.

    """
    # Calculate the cosine similarity between the estimated and
    #true factors
    W = np.zeros((tempW.shape[0],tempW.shape[1]))
    cos = np.zeros((tempW.shape[1], W_true.shape[1]))
    for k in range(W_true.shape[1]):
        for j in range(tempW.shape[1]):
            cos[j,k] = cosine_similarity([W_true[:,k]],[tempW[:,j]])
    sim_factors = np.argmax(np.absolute(cos),axis=0)
    
    # Sort the factors based on the similarity between estimated and
    #true factors.  
    flip = []
    for comp in range(sim_factors.size):
        if cos[sim_factors[comp],comp] > 0:
            W[:,comp] = tempW[:,sim_factors[comp]]
            flip.append(1)
        elif cos[sim_factors[comp],comp] < 0:
            #flip sign of the factor
            W[:,comp] =  - tempW[:,sim_factors[comp]]
            flip.append(-1)
    return W, sim_factors, flip                         

def plot_loadings(W, d, W_path):

    """ 
    Plot loadings.

    Parameters
    ----------
    W : array-like, shape(n_features, n_comps)
        Concatenated loading matrix. The number of features
        here correspond to the total number of features in 
        all groups. 

    d : array-like, shape(n_groups,)
        Number of features in each group.

    W_path : str
        Path to save the figure.     
    
    """
    x = np.linspace(1,3,num=W.shape[0])
    step = 6
    c = step * W.shape[1]+1
    plt.figure()
    for col in range(W.shape[1]):
        y = W[:,col] + (col+c)
        c-=step
        plt.plot( x, y, color='black', linewidth=1.5)
    fig = plt.gca()
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.axvline(x=x[d[0]-1],color='red')   
    plt.savefig(W_path)
    plt.close()

def plot_Z(Z, Z_path, match=False, s_comps=None, flip=None):

    """ 
    Plot latent variables.

    Parameters
    ----------
    Z : array-like, shape(n_features, n_comps)
        Latent variables matrix.

    Z_path : str
        Path to save the figure.

    match : bool, defaults to False.
        Match (or not) the latent components.

    s_comps : array-like, shape(n_comps,), defaults to None.
        Indices to sort the latent components.  

    flip : list, defaults to None.
        Indices to flip the latent components. Positive cosine 
        similarity corresponds to the same sign and negative 
        cosine similarity represents inverse sign.    
    
    """   
    x = np.linspace(0, Z.shape[0], Z.shape[0])
    ncomp = Z.shape[1]
    fig = plt.figure(figsize=(8, 6))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    for j in range(ncomp):
        ax = fig.add_subplot(ncomp, 1, j+1)    
        if match:
            ax.scatter(x, Z[:, s_comps[j]] * flip[j], s=4)
        else:
            ax.scatter(x, Z[:, j], s=4)
        ax.set_xticks([])
        ax.set_yticks([])       
    plt.savefig(Z_path)
    plt.close()           

def plot_params(model, res_dir, args, best_run, data, plot_trueparams=False, plot_medianparams=False):
    
    """ 
    Plot the model parameters and ELBO.

    Parameters
    ----------
    model : Outputs of the model.

    res_dir : str
        Path to the directory where the results will be saved.   

    args : local namespace 
        Arguments selected to run the model.

    best_run : int
        Index of the best model.

    data : dict
        Training and test data, as well as the model parameters 
        used to generate the data.

    plot_trueparams : bool, defaults to False.
        Plot (or not) the model parameters used to generate the 
        data.                
    
    plot_medianparams : bool, defaults to False.
        Plot (or not) the output model parameters when the missing
        values were imputed using the median before training the 
        model. 

    """
    #Concatenate loadings and alphas across data sources    
    W_est = np.zeros((np.sum(model.d),model.k))
    alphas_est = np.zeros((model.k, args.num_sources))
    W_true = np.zeros((np.sum(model.d),data['true_K']))
    if plot_trueparams:
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

    # Plot loading matrices
    if plot_trueparams:
        #plot true Ws
        W_path = f'{res_dir}/[{best_run+1}]W_true.png'
        plot_loadings(W_true, model.d, W_path) 
    
    #plot estimated Ws
    if model.k == data['true_K']:
        #match true and estimated components
        W_est, sim_factors, flip = match_factors(W_est, W_true) 
    if plot_medianparams:                          
        W_path = f'{res_dir}/[{best_run+1}]W_est_median.png'
    else:
        W_path = f'{res_dir}/[{best_run+1}]W_est.png'           
    plot_loadings(W_est, model.d, W_path)
    
    # Plot latent variables
    if plot_trueparams:    
        #plot true latent 
        Z_path = f'{res_dir}/[{best_run+1}]Z_true.png'    
        plot_Z(data['Z'], Z_path)
    #plot estimated latent variables
    if plot_medianparams:                          
        Z_path = f'{res_dir}/[{best_run+1}]Z_est_median.png'
    else:
        Z_path = f'{res_dir}/[{best_run+1}]Z_est.png'
    if model.k == data['true_K']:
        plot_Z(model.means_z, Z_path, match=True, s_comps=sim_factors, flip=flip)
    else:     
        plot_Z(model.means_z, Z_path)       

    # Plot alphas
    if plot_trueparams:
        #plot true alphas
        alphas_path = f'{res_dir}/[{best_run+1}]alphas_true.png'
        hinton_diag(-alphas_true.T, alphas_path)     
    #plot estimated alphas
    if plot_medianparams:                          
        alphas_path = f'{res_dir}/[{best_run+1}]alphas_est_median.png'
    else:
        alphas_path = f'{res_dir}/[{best_run+1}]alphas_est.png'
    if model.k == data['true_K']:
        hinton_diag(-alphas_est[sim_factors,:].T, alphas_path) 
    else:
        hinton_diag(-alphas_est.T, alphas_path)

    # Plot ELBO
    if plot_medianparams:
        L_path = f'{res_dir}/[{best_run+1}]ELBO_median.png'
    else:
        L_path = f'{res_dir}/[{best_run+1}]ELBO.png'    
    plt.figure(figsize=(5, 4))
    plt.plot(model.L[1:])
    plt.savefig(L_path)
    plt.close() 

def get_results(args, res_dir, InfoMiss=None):  

    """ 
    Plot and save the results of the experiments on synthetic data.

    Parameters
    ----------   
    args : local namespace 
        Arguments selected to run the model.

    res_dir : str
        Path to the directory where the results will be saved.

    infoMiss : dict | None, optional.
        Parameters to generate data with missing values.         

    """  

    nruns = args.num_runs #number of runs   
    # Initialise variables to save MSEs, correlations and ELBO values
    MSE = np.zeros((1, nruns))
    MSE_chlv = np.zeros((1, nruns))
    if 'incomplete' in args.scenario:
        Corr_miss = np.zeros((len(InfoMiss['group']), nruns)) 
    if args.impMedian:
        MSEmed = np.zeros((1, nruns))          
    ELBO = np.zeros((1, nruns))    
    ofile = open(f'{res_dir}/results.txt','w')   
    
    for i in range(0, nruns):
        print('\nInitialisation: ', i+1, file=ofile)
        # Load files
        model_file = f'{res_dir}/[{i+1}]ModelOutput.dictionary'
        with open(model_file, 'rb') as parameters:
            GFAotp = pickle.load(parameters)
        if args.impMedian: 
            model_median_file = f'{res_dir}/[{i+1}]ModelOutput_median.dictionary'
            with open(model_median_file, 'rb') as parameters:
                GFAotp_median = pickle.load(parameters)
        
        # Print ELBO 
        ELBO[0,i] = GFAotp.L[-1]
        print('ELBO (last value):', np.around(ELBO[0,i],2), file=ofile)
        # Print estimated taus
        for m in range(GFAotp.s):
            print(f'Estimated avg. taus (data source {m+1}):', np.around(np.mean(GFAotp.E_tau[m]),2), file=ofile)                             
            
        # Get predictions
        MSE[0,i] = GFAotp.MSE
        MSE_chlv[0,i] = GFAotp.MSE_chlev
        if args.impMedian:
            MSEmed[0,i] = GFAotp_median.MSE
        # Get correlation between true and predicted missing values
        if 'incomplete' in args.scenario:
            for j in range(len(InfoMiss['group'])):
                Corr_miss[j,i] = GFAotp.Corr_miss[0,j]                      

    # Plot results for the best run
    #---------------------------------------------------------------------------
    # Get best run
    best_run = int(np.argmax(ELBO))
    print('\nOVERALL RESULTS--------------------------', file=ofile)   
    print('BEST RUN: ', best_run+1, file=ofile)
    # Load model output and data files of the best run
    model_file = f'{res_dir}/[{best_run+1}]ModelOutput.dictionary'
    with open(model_file, 'rb') as parameters:
        GFAotp_best = pickle.load(parameters)
    data_file = f'{res_dir}/[{best_run+1}]Data.dictionary'    
    with open(data_file, 'rb') as parameters:
        data = pickle.load(parameters) 

    # Plot true and estimated parameters
    plot_params(GFAotp_best, res_dir, args, best_run, data, plot_trueparams=True) 

    # Calculate total variance explained
    T = np.zeros((1,np.sum(GFAotp_best.d)))
    W = np.zeros((np.sum(GFAotp_best.d),GFAotp_best.k))
    W_true = np.zeros((np.sum(GFAotp_best.d), data['true_K']))
    d = 0
    for m in range(args.num_sources):
        Dm = GFAotp_best.d[m]
        if 'diagonal' in args.noise:       
            T[0,d:d+Dm] = 1/GFAotp_best.E_tau[m]
        else:
            T[0,d:d+Dm] = np.ones((1,Dm)) * 1/GFAotp_best.E_tau[m]
        W[d:d+Dm,:] = GFAotp_best.means_w[m]
        W_true[d:d+Dm,:] = data['W'][m]
        d += Dm          
    T = np.diag(T[0,:])
    if GFAotp_best.k == data['true_K']:
        #match true and estimated factors
        match_res = match_factors(W, W_true)
        W = match_res[0]
    # Calculate total variance explained    
    Est_totalvar = np.trace(np.dot(W,W.T) + T) 
    
    # Find relevant factors
    relfact_sh, relfact_sp = find_relfactors(W, GFAotp_best, Est_totalvar)
    print('\nTotal variance explained by the true factors: ', np.around(np.trace(np.dot(W_true,W_true.T)),2), file=ofile)
    print('Total variance explained by the estimated factors: ', np.around(np.trace(np.dot(W,W.T)),2), file=ofile)
    print('Relevant shared factors: ', np.array(relfact_sh)+1, file=ofile)
    for m in range(args.num_sources):
        print(f'Relevant specific factors (data source {m+1}): ', np.array(relfact_sp[m])+1, file=ofile)

    # Multi-output predictions
    print('\nMulti-output predictions-----------------',file=ofile)
    print('Model with observed data only:',file=ofile)
    print(f'MSE (avg(std)): {np.around(np.mean(MSE),2)} ({np.around(np.std(MSE),3)})', file=ofile)
    print('\nChance level:',file=ofile)
    print(f'MSE (avg(std)): {np.around(np.mean(MSE_chlv),2)} ({np.around(np.std(MSE),3)})', file=ofile)
    # Missing data prediction
    if 'incomplete' in args.scenario:
        print('\nPredictions for missing data -----------------',file=ofile)
        for j in range(len(InfoMiss['group'])):
            print('Data source: ',InfoMiss['group'][j], file=ofile)
            print(f'Correlation (avg(std)): {np.around(np.mean(Corr_miss[j,:]),3)} ({np.around(np.std(Corr_miss[j,:]),3)})', file=ofile)    

    # Plot model parameters obtained with the complete data (using median imputation) 
    if args.impMedian:
        print('\nModel with median imputation------------',file=ofile)
        #Load model output and data files of the best run
        model_file = f'{res_dir}/[{best_run+1}]ModelOutput_median.dictionary'
        with open(model_file, 'rb') as parameters:
            GFAotp_median_best = pickle.load(parameters)
        
        # Print taus
        for m in range(args.num_sources):
            print(f'Estimated avg. taus (data source {m+1}):', np.around(np.mean(GFAotp_median_best.E_tau[m]),2), file=ofile)
        # Plot estimated parameters
        plot_params(GFAotp_median_best, res_dir, args, best_run, data, plot_medianparams=True)
        # Print predictions
        print('Predictions:',file=ofile)
        print(f'MSE (avg(std)): {np.around(np.mean(MSEmed),3)} ({np.around(np.std(MSEmed),3)})', file=ofile) 

    print('Visualisation concluded!')                               
    ofile.close() 
        

        
   