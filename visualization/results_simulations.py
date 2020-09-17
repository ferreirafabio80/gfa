import numpy as np
import matplotlib.pyplot as plt
import pickle
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

def find_relfactors(W, model, total_var, res_dir):
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

def plot_params(model, res_dir, args, best_run, data, plot_trueparams=True, plot_median=False):
    #Concatenate parameters across data sources    
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

    #LOADING MATRICES
    if plot_trueparams:
        #plot true Ws
        W_path = f'{res_dir}/[{best_run+1}]W_true.png'
        plot_weights(W_true, model.d, W_path) 
    #plot estimated Ws
    if model.k == data['true_K']:
        #match true and estimated components
        W_est, comp_e, flip = match_comps(W_est, W_true) 
    if plot_median:                          
        W_path = f'{res_dir}/[{best_run+1}]W_est_median.png'
    else:
        W_path = f'{res_dir}/[{best_run+1}]W_est.png'           
    plot_weights(W_est, model.d, W_path)
    
    #LATENT VARIABLES
    if plot_trueparams:    
        #plot true latent 
        Z_path = f'{res_dir}/[{best_run+1}]Z_true.png'    
        plot_Z(data['Z'], path=Z_path, match=False)
    #plot estimated latent variables
    if plot_median:                          
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
    if plot_median:                          
        alphas_path = f'{res_dir}/[{best_run+1}]alphas_est_median.png'
    else:
        alphas_path = f'{res_dir}/[{best_run+1}]alphas_est.png'
    if model.k == data['true_K']:
        hinton_diag(-alphas_est[comp_e,:].T, alphas_path) 
    else:
        hinton_diag(-alphas_est.T, alphas_path)

    #plot ELBO
    if plot_median:
        L_path = f'{res_dir}/[{best_run+1}]ELBO_median.png'
    else:
        L_path = f'{res_dir}/[{best_run+1}]ELBO.png'    
    plt.figure(figsize=(5, 4))
    plt.plot(model.L[1:])
    plt.savefig(L_path)
    plt.close() 

def get_results(args, res_dir, InfoMiss=None):    

    nruns = args.num_runs #number of runs   
    #initialise variables to save MSEs, correlations and ELBO values
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
        print('ELBO (last value):', np.around(ELBO[0,i],2), file=ofile)
        #Print estimated taus
        for m in range(GFAotp.s):
            print(f'Estimated avg. taus (data source {m+1}):', np.around(np.mean(GFAotp.E_tau[m]),2), file=ofile)                             
            
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
    print('BEST RUN: ', best_run+1, file=ofile)
    #Load model output and data files of the best run
    model_file = f'{res_dir}/[{best_run+1}]ModelOutput.dictionary'
    with open(model_file, 'rb') as parameters:
        GFAotp_best = pickle.load(parameters)
    data_file = f'{res_dir}/[{best_run+1}]Data.dictionary'    
    with open(data_file, 'rb') as parameters:
        data = pickle.load(parameters) 

    #Plot true and estimated parameters
    plot_params(GFAotp_best, res_dir, args, best_run, data=data) 

    #Compute total variance explained
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
        #match true and estimated components
        match_res = match_comps(W, W_true)
    W = match_res[0]    
    Est_totalvar = np.trace(np.dot(W,W.T) + T) 
    
    #Find relevant factors
    relfact_sh, relfact_sp = find_relfactors(W, GFAotp_best, Est_totalvar, res_dir)
    print('\nTotal variance explained by the true factors: ', np.around(np.trace(np.dot(W_true,W_true.T)),2), file=ofile)
    print('Total variance explained by the estimated factors: ', np.around(np.trace(np.dot(W,W.T)),2), file=ofile)
    print('Relevant shared factors: ', np.array(relfact_sh)+1, file=ofile)
    for m in range(args.num_sources):
        print(f'Relevant specific factors (data source {m+1}): ', np.array(relfact_sp[m])+1, file=ofile)

    #Multi-output predictions
    #--------------------------------------------------------------------------------
    print('\nMulti-output predictions-----------------',file=ofile)
    print('Model with observed data only:',file=ofile)
    print(f'MSE (avg(std)): {np.around(np.mean(MSE),2)} ({np.around(np.std(MSE),2)})', file=ofile)
    print('\nChance level:',file=ofile)
    print(f'MSE (avg(std)): {np.around(np.mean(MSE_chlv),2)} ({np.around(np.std(MSE),2)})', file=ofile)
    #Missing data prediction
    if 'incomplete' in args.scenario:
        print('\nPredictions for missing data -----------------',file=ofile)
        for j in range(len(InfoMiss['group'])):
            print('Data source: ',j+1, file=ofile)
            print(f'Correlation (avg(std)): {np.around(np.mean(Corr_miss[j,:]),2)} ({np.around(np.std(Corr_miss[j,:]),2)})', file=ofile)    

    if args.impMedian:
        print('\nModel with median imputation------------',file=ofile)
        #Load model output and data files of the best run
        model_file = f'{res_dir}/[{best_run+1}]ModelOutput_median.dictionary'
        with open(model_file, 'rb') as parameters:
            GFAotp_median_best = pickle.load(parameters)
        
        #Plot estimated parameters
        #taus
        for m in range(args.num_sources):
            print(f'Estimated avg. taus (data source {m+1}):', np.around(np.mean(GFAotp_median_best.E_tau[m]),2), file=ofile)
        #W, Z and ELBO
        plot_params(GFAotp_median_best, res_dir, args, best_run, data, plot_trueparams=False, plot_median=True)
        #predictions
        print('Predictions:',file=ofile)
        print(f'MSE (avg(std)): {np.around(np.mean(MSEmed),2)} ({np.around(np.std(MSEmed),2)})', file=ofile) 

    print('Visualisation concluded!')                               
    ofile.close() 
        

        
   