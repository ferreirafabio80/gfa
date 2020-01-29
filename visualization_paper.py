import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import xlsxwriter
import plotly.graph_objects as go
import os
from scipy import io
from utils import GFAtools
from scipy.stats import multivariate_normal
from sklearn.metrics.pairwise import cosine_similarity

def hinton(matrix, path, fcolor, max_weight=None, ax=None):

    # Draw Hinton diagram for visualizing a weight matrix.
    ax = ax if ax is not None else plt.gca()

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

def results_HCP(exp_dir, data_dir):

    filepath = f'{exp_dir}GFA_results.dictionary'
    #Load file
    with open(filepath, 'rb') as parameters:
        res = pickle.load(parameters)

    for i in range(0, len(res)):
        #checking NaNs
        if 'missing' in filepath:
            if 'view1' in filepath:
                total = res[i].X_nan[0].size
                n_miss = np.flatnonzero(res[i].X_nan[0]).shape[0]
                print('Percentage of missing data: ', round((n_miss/total)*100))
            else:
                total = res[i].X_nan[1].size
                n_miss = np.flatnonzero(res[i].X_nan[1]).shape[0]
                print('Percentage of missing data: ', round((n_miss/total)*100))

        if 'training' in filepath:
            N_train = res[i].N
            N_test = res[i].X_test[0].shape[0]
            print('Percentage of train data: ', round(N_train/(N_test+N_train)*100))

        #Computational time
        print('Computational time (hours): ', round(res[i].time_elapsed/3600))

        #Weights and total variance
        W1 = res[i].means_w[0]
        W2 = res[i].means_w[1]
        W = np.concatenate((W1, W2), axis=0)        
        if 'PCA' in filepath:
            S1 = res[i].E_tau[0] * np.ones((1, W1.shape[0]))[0]
            S2 = res[i].E_tau[1] * np.ones((1, W2.shape[0]))[0]
            S = np.diag(np.concatenate((S1, S2), axis=0))
        else:
            S1 = res[i].E_tau[0]
            S2 = res[i].E_tau[1]
            S = np.diag(np.concatenate((S1, S2), axis=1)[0,:])
        total_var = np.trace(np.dot(W,W.T) + S)    
        
        #Explained variance
        var1 = np.zeros((1, W1.shape[1])) 
        var2 = np.zeros((1, W1.shape[1])) 
        var = np.zeros((1, W.shape[1]))
        for c in range(0, W.shape[1]):
            w = np.reshape(W[:,c],(W.shape[0],1))
            w1 = np.reshape(W1[:,c],(W1.shape[0],1))
            w2 = np.reshape(W2[:,c],(W2.shape[0],1))
            var1[0,c] = (np.trace(np.dot(w1.T, w1))/total_var) * 100
            var2[0,c] = (np.trace(np.dot(w2.T, w2))/total_var) * 100
            var[0,c] = (np.trace(np.dot(w.T, w))/total_var) * 100

        var_path = f'{exp_dir}/variances{i+1}.xlsx'
        df = pd.DataFrame({'components':range(1, W.shape[1]+1),'Brain': list(var1[0,:]),'Behaviour': list(var2[0,:]), 'Both': list(var[0,:])})
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(var_path, engine='xlsxwriter')
        # Convert the dataframe to an XlsxWriter Excel object.
        df.to_excel(writer, sheet_name='Sheet1')
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()

        relvar_path = f'{exp_dir}/relative_variances{i+1}.xlsx'
        relvar1 = np.zeros((1, W1.shape[1])) 
        relvar2 = np.zeros((1, W1.shape[1]))
        relvar = np.zeros((1, W1.shape[1]))
        for j in range(0, W.shape[1]):
            relvar1[0,j] = 100 - ((np.sum(var1[0,:]) - var1[0,j])/np.sum(var1[0,:])) * 100 
            relvar2[0,j] = 100 - ((np.sum(var2[0,:]) - var2[0,j])/np.sum(var2[0,:])) * 100  
            relvar[0,j] = 100 - ((np.sum(var[0,:]) - var[0,j])/np.sum(var[0,:])) * 100  

        df = pd.DataFrame({'components':range(1, W.shape[1]+1),'Brain': list(relvar1[0,:]),'Behaviour': list(relvar2[0,:]), 'Both': list(relvar[0,:])})
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer1 = pd.ExcelWriter(relvar_path, engine='xlsxwriter')
        # Convert the dataframe to an XlsxWriter Excel object.
        df.to_excel(writer1, sheet_name='Sheet1')
        # Close the Pandas Excel writer and output the Excel file.
        writer1.save()         

        #Select shared and specific components
        ind1 = []
        ind2 = []
        shvar = 1.5
        spvar = 10
        for j in range(0, W.shape[1]):
            if relvar[0,j] > shvar and relvar1[0,j] > shvar and relvar2[0,j] > shvar:
                #shared component
                ind1.append(j) 
                ind2.append(j) 
            elif relvar[0,j] > shvar and relvar1[0,j] > spvar:
                #brain-specific component
                ind1.append(j) 
            elif relvar[0,j] > shvar and relvar2[0,j] > spvar:
                #behaviour-specific component
                ind2.append(j)                  
        
        #Components
        print('Brain components: ', ind1)
        print('Clinical components: ', ind2)

        #Clinical weights
        brain_weights = {"wx": W1[:,np.array(ind1)]}
        io.savemat(f'{exp_dir}/wx{i+1}.mat', brain_weights)
        #Brain weights
        clinical_weights = {"wy": W2[:,np.array(ind2)]}
        io.savemat(f'{exp_dir}/wy{i+1}.mat', clinical_weights)

        #Plot lower bound
        L_path = f'{exp_dir}/LB{i+1}.png'
        plt.figure()
        plt.title('Lower Bound')
        plt.plot(res[i].L[1:])
        plt.savefig(L_path)
        plt.close()

        #-Predictions 
        #---------------------------------------------------------------------
        #Predict missing values
        if 'training' in filepath:   
            
            if 'missing' in filepath:
                if 'view1' in filepath:
                    obs_view = np.array([0, 1])
                    mask_miss = res[i].X_nan[0]==1
                    X = res[i].X_train        
                    missing_true = np.where(mask_miss,X[0],0)       
                    X[0][mask_miss] = 'NaN'
                    missing_pred = GFAtools(X, res[i], obs_view).PredictMissing()
                    miss_true = missing_true[mask_miss]
                    miss_pred = missing_pred[0][mask_miss]
                elif 'view2' in filepath:
                    obs_view = np.array([1, 0])
                    mask_miss = res[i].X_nan[1]==1
                    X = res[i].X_train        
                    missing_true = np.where(mask_miss, X[1],0)       
                    X[1][mask_miss] = 'NaN'
                    missing_pred = GFAtools(X, res[i], obs_view).PredictMissing()
                    miss_true = missing_true[mask_miss]
                    miss_pred = missing_pred[0][mask_miss]    
                MSEmissing = np.mean((miss_true - miss_pred) ** 2)
                print('MSE for missing data: ', MSEmissing)

        """ obs_view1 = np.array([0, 1])
        obs_view2 = np.array([1, 0])
        vpred1 = np.array(np.where(obs_view1 == 0))
        vpred2 = np.array(np.where(obs_view2 == 0))
        X_pred = [[] for _ in range(res[i].d.size)]
        sig_pred = [[] for _ in range(res[i].d.size)]
        X_predmean = [[] for _ in range(res[i].d.size)]
        X_pred[vpred1[0,0]], sig_pred[vpred1[0,0]] = GFAtools(res[i].X_test, res[i], obs_view1).PredictView(noise)
        X_pred[vpred2[0,0]], sig_pred[vpred2[0,0]] = GFAtools(res[i].X_test, res[i], obs_view2).PredictView(noise)
        meanX = np.array((np.mean(X[0]),np.mean(X[1])))
        Ntest = res[i].X_test[0].shape[0] 
        X_predmean[vpred1[0,0]] = meanX[vpred1[0,0]] * np.ones((Ntest,res[i].d[vpred1[0,0]]))
        X_predmean[vpred2[0,0]] = meanX[vpred2[0,0]] * np.ones((Ntest,res[i].d[vpred2[0,0]]))

        #-Metrics
        #----------------------------------------------------------------------------------
        probs = [np.zeros((1,res[i].X_test[0].shape[0])) for _ in range(res[i].d.size)]
        for j in range(res[i].X_test[0].shape[0]):
            probs[vpred1[0,0]][0,j] = multivariate_normal.pdf(res[i].X_test[vpred1[0,0]][j,:], 
                mean=X_pred[vpred1[0,0]][j,:], cov=sig_pred[vpred1[0,0]])
            #probs[vpred2[0,0]][0,j] = multivariate_normal.pdf(res[i].X_test[vpred2[0,0]][j,:], 
            #    mean=X_pred[vpred2[0,0]][j,:], cov=sig_pred[vpred2[0,0]])

        #sum_probs = np.sum(probs[0])

        A1 = res[i].X_test[vpred1[0,0]] - X_pred[vpred1[0,0]]
        A2 = res[i].X_test[vpred1[0,0]] - X_predmean[vpred1[0,0]]
        Fnorm1 = np.sqrt(np.trace(np.dot(A1,A1.T)))
        Fnorm_mean1 = np.sqrt(np.trace(np.dot(A2,A2.T)))

        A1 = res[i].X_test[vpred2[0,0]] - X_pred[vpred2[0,0]]
        A2 = res[i].X_test[vpred2[0,0]] - X_predmean[vpred2[0,0]]
        Fnorm2 = np.sqrt(np.trace(np.dot(A1,A1.T)))
        Fnorm_mean2 = np.sqrt(np.trace(np.dot(A2,A2.T))) """ 

def results_simulations(exp_dir):
    
    #Load file
    filepath = f'{exp_dir}/GFA_results.dictionary'
    with open(filepath, 'rb') as parameters:
        res = pickle.load(parameters)
    
    Lower_bounds = np.zeros((1,len(res)))
    MSE = np.zeros((1,len(res)))
    for i in range(0, len(res)):

        print('Initialization: ', i+1)    
        #checking NaNs
        if 'missing' in filepath:
            for j in range(len(res[i].remove)):
                v_miss = res[i].vmiss[j] 
                total = res[i].X_nan[v_miss-1].size
                n_miss = np.flatnonzero(res[i].X_nan[v_miss-1]).shape[0]
                print(f'Percentage of missing data of view {v_miss}: ', round((n_miss/total)*100))

        if 'training' in filepath:
            N_train = res[i].N
            N_test = res[i].N_test
            print('Percentage of train data: ', round(N_train/(N_test+N_train)*100))

        Lower_bounds[0,i] = res[i].L[-1] 

        if 'training' in filepath:
            if 'missing' in filepath:
                file_missing = f'{exp_dir}/GFA_results_imputation.dictionary'
                with open(file_missing, 'rb') as parameters:
                    res1 = pickle.load(parameters)

                file_median = f'{exp_dir}/GFA_results_median.dictionary'
                with open(file_median, 'rb') as parameters:
                    res2 = pickle.load(parameters) 

                print("MSE missing data:", res[i].MSEmissing)
                MSE[0,i] = res[i].MSEmissing          
    
            #Plot predictions
            #--------------------------------------------------------------------------------------------------------
            obs_view = np.array([0, 1])
            #view 1 from view 2
            vpred1 = np.where(obs_view == 0)
            if 'missing' in filepath:
                df = pd.DataFrame(columns=['x', 'Pred_nomissing','Pred_imputation','Pred_mean'])
                for j in range(res[i].d[vpred1[0][0]]):
                    df = df.append({'x':j+1, 'Pred_nomissing': res[i].reMSE1[0,j], 
                    'Pred_imputation': res1[i].reMSE1[0,j], 'Pred_median': res2[i].reMSE1[0,j], 
                    'Pred_mean': res[i].reMSEmean1[0,j]}, ignore_index=True)
                ymax = max(np.max(res[i].reMSE1),np.max(res1[i].reMSE1), 
                np.max(res2[i].reMSE1[0,j]), np.max(res[i].reMSEmean1))
                title = 'Predict view 1 from view 2 (incomplete)'    
            else:
                df = pd.DataFrame(columns=['x', 'Pred_nomissing','Pred_mean'])
                for j in range(res[i].d[vpred1[0][0]]):
                    df = df.append({'x':j+1, 'Pred_nomissing': res[i].reMSE1[0,j], 
                        'Pred_mean': res[i].reMSEmean1[0,j]}, ignore_index=True)
                ymax = max(np.max(res[i].reMSE1), np.max(res[i].reMSEmean1))         
                title = 'Predict view 1 from view 2 (complete)'
            line_path = f'{exp_dir}/predictions_view1_{i+1}.svg'         
            plot_predictions(df, ymax, title, line_path)

            #view 2 from view 1
            vpred2 = np.where(obs_view == 1)
            if 'missing' in filepath:
                df = pd.DataFrame(columns=['x', 'Pred_nomissing','Pred_imputation','Pred_mean'])
                for j in range(res[i].d[vpred2[0][0]]):
                    df = df.append({'x':j+1, 'Pred_nomissing': res[i].reMSE2[0,j], 
                    'Pred_imputation': res1[i].reMSE2[0,j], 'Pred_median': res2[i].reMSE2[0,j], 
                    'Pred_mean': res[i].reMSEmean2[0,j]}, ignore_index=True)
                ymax = max(np.max(res[i].reMSE2),np.max(res1[i].reMSE2), 
                np.max(res2[i].reMSE2[0,j]), np.max(res[i].reMSEmean2))
                title = 'Predict view 2 from view 1 (incomplete)'
            else:
                df = pd.DataFrame(columns=['x', 'Pred_nomissing','Pred_mean'])
                for j in range(res[i].d[vpred2[0][0]]):
                    df = df.append({'x':j+1, 'Pred_nomissing': res[i].reMSE2[0,j], 
                        'Pred_mean': res[i].reMSEmean2[0,j]}, ignore_index=True)
                title = 'Predict view 2 from view 1 (complete)'
                ymax = max(np.max(res[i].reMSE2), np.max(res[i].reMSEmean2))                 
            line_path = f'{exp_dir}/predictions_view2_{i+1}.svg'
            plot_predictions(df, ymax, title, line_path) 
 

        # plot true projections
        W1 = res[i].W[0]
        W2 = res[i].W[1]
        W_true = np.concatenate((W1, W2), axis=0)
        W_true[np.absolute(W_true) < 0.005] = 0
        
        if 'lowD' in filepath:
            W_path = f'{exp_dir}/true_W1_{i+1}.svg'
            color = 'gray'
            fig = plt.figure()
            hinton(W_true, W_path, color)
        
        #plot estimated projections
        W1 = res[i].means_w[0]
        W2 = res[i].means_w[1]
        tempW = np.concatenate((W1, W2), axis=0)
        W = np.zeros((tempW.shape[0],tempW.shape[1]))
        cos = np.zeros((tempW.shape[1], W_true.shape[1]))
        for k in range(W_true.shape[1]):
            for j in range(tempW.shape[1]):
                cos[j,k] = cosine_similarity([W_true[:,k]],[tempW[:,j]])
        comp_e = np.argmax(np.absolute(cos),axis=0)
        max_cos = np.max(np.absolute(cos),axis=0)
        comp_e = comp_e[max_cos > 0.7] 
        flip = []       
        for comp in range(comp_e.size):
            if cos[comp_e[comp],comp] > 0:
                W[:,comp] = tempW[:,comp_e[comp]]
                flip.append(1)
            elif cos[comp_e[comp],comp] < 0:
                W[:,comp] =  - tempW[:,comp_e[comp]]
                flip.append(-1)
        if 'lowD' in filepath:                  
            W_path = f'{exp_dir}/estimated_W_{i+1}.svg'
            W[np.absolute(W) < 0.005] = 0       
            fig = plt.figure()
            hinton(W, W_path, color)

        # plot estimated latent variables
        Z_path = f'{exp_dir}/estimated_Z_{i+1}.svg'
        x = np.linspace(0, res[i].means_z.shape[0], res[i].means_z.shape[0])
        numsub = res[i].means_z.shape[1]
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for j in range(numsub):
            ax = fig.add_subplot(numsub, 1, j+1)
            ax.scatter(x, res[i].means_z[:, comp_e[j]] * flip[j])
        plt.savefig(Z_path)
        plt.close()

        # plot true latent variables
        Z_path = f'{exp_dir}/true_Z_{i+1}.svg'
        x = np.linspace(0, res[i].Z.shape[0], res[i].Z.shape[0])
        numsub = res[i].Z.shape[1]
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for j in range(1, numsub+1):
            ax = fig.add_subplot(numsub, 1, j)
            ax.scatter(x, res[i].Z[:, j-1])
        plt.savefig(Z_path)
        plt.close()

        #plot estimated alphas
        a_path = f'{exp_dir}/estimated_alphas_{i+1}.svg'
        color = 'white'
        a1 = np.reshape(res[i].E_alpha[0], (res[i].k, 1))
        a2 = np.reshape(res[i].E_alpha[1], (res[i].k, 1))
        a = np.concatenate((a1, a2), axis=1)
        fig = plt.figure()
        hinton(-a[comp_e,:].T, a_path, color) 

        #plot true alphas
        a_path = f'{exp_dir}/true_alphas_{i+1}.svg'
        a1 = np.reshape(res[i].alphas[0], (res[i].alphas[0].shape[0], 1))
        a2 = np.reshape(res[i].alphas[1], (res[i].alphas[1].shape[0], 1))
        a = np.concatenate((a1, a2), axis=1)
        fig = plt.figure()
        hinton(-a.T, a_path, color)        

        # plot lower bound
        L_path = f'{exp_dir}/LB_{i+1}.svg'
        fig = plt.figure()
        plt.plot(res[i].L[1:])
        plt.savefig(L_path)
        plt.close()

    if 'training' in filepath:
        best_init = int(np.argmin(MSE)+1)
    else:
        best_init = int(np.argmax(Lower_bounds)+1)    
    print("Best initialization: ", best_init)
    np.savetxt(f'{exp_dir}/best_init.txt', np.atleast_1d(best_init))       

    
    #plot estimated projections
    W1 = res1[best_init-1].means_w[0]
    W2 = res1[best_init-1].means_w[1]
    W = np.concatenate((W1, W2), axis=0)
    #W = np.zeros((tempW.shape[0],tempW.shape[1]))
    """ cos = np.zeros((tempW.shape[1], W_true.shape[1]))
    for k in range(W_true.shape[1]):
        for j in range(tempW.shape[1]):
            cos[j,k] = cosine_similarity([W_true[:,k]],[tempW[:,j]])
    comp_e = np.argmax(np.absolute(cos),axis=0)
    max_cos = np.max(np.absolute(cos),axis=0)
    comp_e = comp_e[max_cos > 0.7] 
    flip = []       
    for comp in range(comp_e.size):
        if cos[comp_e[comp],comp] > 0:
            W[:,comp] = tempW[:,comp_e[comp]]
            flip.append(1)
        elif cos[comp_e[comp],comp] < 0:
            W[:,comp] =  - tempW[:,comp_e[comp]]
            flip.append(-1) """
    if 'lowD' in filepath:                  
        W_path = f'{exp_dir}/estimated_W_MODEL2.svg'
        W[np.absolute(W) < 0.005] = 0       
        fig = plt.figure()
        color = 'grey'
        hinton(W, W_path, color)

    # plot estimated latent variables
    Z_path = f'{exp_dir}/estimated_Z_MODEL2.svg'
    x = np.linspace(0, res1[best_init-1].means_z.shape[0], res1[best_init-1].means_z.shape[0])
    numsub = res1[best_init-1].means_z.shape[1]
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for j in range(numsub):
        ax = fig.add_subplot(numsub, 1, j+1)
        ax.scatter(x, res1[best_init-1].means_z[:,j] * flip[j])
    plt.savefig(Z_path)
    plt.close()   

    #plot estimated projections
    W1 = res2[best_init-1].means_w[0]
    W2 = res2[best_init-1].means_w[1]
    W = np.concatenate((W1, W2), axis=0)
    #W = np.zeros((tempW.shape[0],tempW.shape[1]))
    """ cos = np.zeros((tempW.shape[1], W_true.shape[1]))
    for k in range(W_true.shape[1]):
        for j in range(tempW.shape[1]):
            cos[j,k] = cosine_similarity([W_true[:,k]],[tempW[:,j]])
    comp_e = np.argmax(np.absolute(cos),axis=0)
    max_cos = np.max(np.absolute(cos),axis=0)
    comp_e = comp_e[max_cos > 0.7] 
    flip = []       
    for comp in range(comp_e.size):
        if cos[comp_e[comp],comp] > 0:
            W[:,comp] = tempW[:,comp_e[comp]]
            flip.append(1)
        elif cos[comp_e[comp],comp] < 0:
            W[:,comp] =  - tempW[:,comp_e[comp]]
            flip.append(-1) """
    if 'lowD' in filepath:                  
        W_path = f'{exp_dir}/estimated_W_MODEL3.svg'
        W[np.absolute(W) < 0.005] = 0       
        fig = plt.figure()
        hinton(W, W_path, color)

    # plot estimated latent variables
    Z_path = f'{exp_dir}/estimated_Z_MODEL3.svg'
    x = np.linspace(0, res2[best_init-1].means_z.shape[0], res2[best_init-1].means_z.shape[0])
    numsub = res2[best_init-1].means_z.shape[1]
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for j in range(numsub):
        ax = fig.add_subplot(numsub, 1, j+1)
        ax.scatter(x, res2[best_init-1].means_z[:,j] * flip[j])
    plt.savefig(Z_path)
    plt.close()   

                    
        

        
   