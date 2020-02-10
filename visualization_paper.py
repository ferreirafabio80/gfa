import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import xlsxwriter
import plotly.graph_objects as go
import os
import statistics as stats
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

def compute_variances(W, d, total_var, shvar, spvar, var_path,relvar_path):

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

    df = pd.DataFrame({'components':range(1, W.shape[1]+1),'Brain': list(var1[0,:]),'Behaviour': list(var2[0,:]), 'Both': list(var[0,:])})
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(var_path, engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

    relvar1 = np.zeros((1, W.shape[1])) 
    relvar2 = np.zeros((1, W.shape[1]))
    relvar = np.zeros((1, W.shape[1]))
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

    return ind1,ind2

def match_comps(tempW,W_true):
    W = np.zeros((tempW.shape[0],tempW.shape[1]))
    cos = np.zeros((tempW.shape[1], W_true.shape[1]))
    for k in range(W_true.shape[1]):
        for j in range(tempW.shape[1]):
            cos[j,k] = cosine_similarity([W_true[:,k]],[tempW[:,j]])
    comp_e = np.argmax(np.absolute(cos),axis=0)
    max_cos = np.max(np.absolute(cos),axis=0)
    comp_e = comp_e[max_cos > 0.65] 
    flip = []       
    for comp in range(comp_e.size):
        if cos[comp_e[comp],comp] > 0:
            W[:,comp] = tempW[:,comp_e[comp]]
            flip.append(1)
        elif cos[comp_e[comp],comp] < 0:
            W[:,comp] =  - tempW[:,comp_e[comp]]
            flip.append(-1)
    return W, comp_e, flip                         

def plot_weights(W, W_path):
    x = np.linspace(1,3,num=W.shape[0])
    df=pd.DataFrame({'x': x})
    step = 6
    c = step * W.shape[1]+1
    plt.figure(figsize=(2.5, 2))
    for col in range(W.shape[1]):
        df[f'y{col+1}'] = W[:,col] + (col+c)
        c-=step
        # multiple line plot
        plt.plot( 'x', f'y{col+1}', data=df, color='black', linewidth=1.5)
    plt.savefig(W_path)
    plt.close()

def results_HCP(ninit, exp_dir, data_dir):

    print("Plot results ------")
    Lower_bounds = np.zeros((1,ninit))
    reMSE = np.zeros((2,ninit))
    for i in range(ninit):
        
        print("Run:", i+1)
        
        filepath = f'{exp_dir}GFA_results{i+1}.dictionary'
        #Load file
        with open(filepath, 'rb') as parameters:
            res = pickle.load(parameters)    

        #checking NaNs
        if 'missing' in filepath:
            if 'view1' in filepath:
                total = res.X_nan[0].size
                n_miss = np.flatnonzero(res.X_nan[0]).shape[0]
                print('Percentage of missing data: ', round((n_miss/total)*100))
            else:
                total = res.X_nan[1].size
                n_miss = np.flatnonzero(res.X_nan[1]).shape[0]
                print('Percentage of missing data: ', round((n_miss/total)*100))

        if 'training' in filepath:
            N_train = res.N
            N_test = res.X_test[0].shape[0]
            print('Percentage of train data: ', round(N_train/(N_test+N_train)*100))

        #Computational time
        print('Computational time (hours): ', round(res.time_elapsed/3600))

        #Weights and total variance
        W1 = res.means_w[0]
        W2 = res.means_w[1]
        W = np.concatenate((W1, W2), axis=0)        
        if 'PCA' in filepath:
            noise = 'PCA'
            S1 = 1/res.E_tau[0] * np.ones((1, W1.shape[0]))[0]
            S2 = 1/res.E_tau[1] * np.ones((1, W2.shape[0]))[0]
            S = np.diag(np.concatenate((S1, S2), axis=0))
        else:
            noise= 'FA'
            S1 = 1/res.E_tau[0]
            S2 = 1/res.E_tau[1]
            S = np.diag(np.concatenate((S1, S2), axis=1)[0,:])
        total_var = np.trace(np.dot(W,W.T) + S) 

        #Compute variances
        shvar = 1
        spvar = 10
        var_path = f'{exp_dir}/variances{i+1}.xlsx'
        relvar_path = f'{exp_dir}/relative_variances{i+1}.xlsx'
        ind1, ind2 = compute_variances(W, res.d,total_var, shvar, spvar, var_path,relvar_path)                   
        
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
        plt.plot(res.L[1:])
        plt.savefig(L_path)
        plt.close()

        #-Predictions 
        #---------------------------------------------------------------------
        #Predict missing values
        if 'training' in filepath:   
            
            if 'missing' in filepath:
                if 'view1' in filepath:
                    obs_view = np.array([0, 1])
                    mask_miss = res.X_nan[0]==1
                    X = res.X_train        
                    missing_true = np.where(mask_miss,X[0],0)       
                    X[0][mask_miss] = 'NaN'
                    missing_pred = GFAtools(X, res, obs_view).PredictMissing()
                    miss_true = missing_true[mask_miss]
                    miss_pred = missing_pred[0][mask_miss]
                elif 'view2' in filepath:
                    obs_view = np.array([1, 0])
                    mask_miss = res.X_nan[1]==1
                    X = res.X_train        
                    missing_true = np.where(mask_miss, X[1],0)       
                    X[1][mask_miss] = 'NaN'
                    missing_pred = GFAtools(X, res, obs_view).PredictMissing()
                    miss_true = missing_true[mask_miss]
                    miss_pred = missing_pred[0][mask_miss]    
                MSEmissing = np.mean((miss_true - miss_pred) ** 2)
                print('MSE for missing data: ', MSEmissing)
        
            obs_view = np.array([1, 0])
            vpred = np.array(np.where(obs_view == 0))
            X_pred = GFAtools(res.X_test, res, obs_view).PredictView(noise)

            #-Metrics
            #----------------------------------------------------------------------------------
            beh_var = np.array((92, 95))
            #relative MSE for each dimension - predict view 2 from view 1
            for j in range(0, beh_var.size):
                reMSE[j,i] = np.mean((res.X_test[vpred[0,0]][:,beh_var[j]] - X_pred[:,beh_var[j]]) ** 2)/  \
                np.mean(res.X_test[vpred[0,0]][:,beh_var[j]] ** 2)

            print('relative MSE - IQ: ', reMSE[0,i])   
            print('relative MSE - Pic Voc test: ', reMSE[1,i])
    
    if 'training' in filepath:
        #best_init = int(np.argmin(MSE)+1)
        print('Mean rMSE(IQ): ', np.mean(reMSE[0,:]))
        print('Std rMSE(IQ): ', np.std(reMSE[0,:]))
    best_init = int(np.argmax(Lower_bounds)+1)    
    print("Best initialization: ", best_init)
    np.savetxt(f'{exp_dir}/best_init.txt', np.atleast_1d(best_init))    

def results_simulations(exp_dir):
    
    #Load file
    filepath = f'{exp_dir}/GFA_results.dictionary'
    with open(filepath, 'rb') as parameters:
        res = pickle.load(parameters)

    #Output file
    ofile = open(f'{exp_dir}/output.txt','w')    
    
    Lower_bounds = np.zeros((1,len(res)))
    if 'missing' in filepath:
        MSE_miss = np.zeros((len(res[0].vmiss), len(res)))
    file_ext = '.png'
    for i in range(0, len(res)):

        print('\nInitialisation: ', i+1, file=ofile)
        print('------------------------------------------------', file=ofile)    
        #checking NaNs
        if 'missing' in filepath:
            for j in range(len(res[i].remove)):
                v_miss = res[i].vmiss[j] 
                total = res[i].X_nan[v_miss-1].size
                n_miss = np.flatnonzero(res[i].X_nan[v_miss-1]).shape[0]
                #print(f'Percentage of missing data of view {v_miss}: ', round((n_miss/total)*100))
                print(f'Percentage of missing data in view {v_miss}: {round((n_miss/total)*100)}', file=ofile)

                print(f'MSE missing data (view {res[i].vmiss[j]}):', res[i].MSEmissing[j], file=ofile)
                MSE_miss[j,i] = res[i].MSEmissing[j]

        if 'training' in filepath:
            N_train = res[i].N
            N_test = res[i].N_test
            print('Percentage of train data: ', round(N_train/(N_test+N_train)*100), file=ofile)

        Lower_bounds[0,i] = res[i].L[-1] 

        if 'training' in filepath:
            if 'missing' in filepath:
                file_missing = f'{exp_dir}/GFA_results_imputation.dictionary'
                with open(file_missing, 'rb') as parameters:
                    res1 = pickle.load(parameters)

                file_median = f'{exp_dir}/GFA_results_median.dictionary'
                with open(file_median, 'rb') as parameters:
                    res2 = pickle.load(parameters)           
    
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
            line_path = f'{exp_dir}/predictions_view1_{i+1}{file_ext}'         
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
            line_path = f'{exp_dir}/predictions_view2_{i+1}{file_ext}'
            plot_predictions(df, ymax, title, line_path) 
 

        # plot true projections
        W1 = res[i].W[0]
        W2 = res[i].W[1]
        W_true = np.concatenate((W1, W2), axis=0)
        if 'lowD' in filepath:
            W_path = f'{exp_dir}/true_W1_{i+1}{file_ext}'
            color = 'gray'
            fig = plt.figure()
            plot_weights(W_true, W_path)            

        #Compute true variances       
        S1 = 1/res[i].tau[0]
        S2 = 1/res[i].tau[1]
        S = np.diag(np.concatenate((S1, S2), axis=0))
        total_var = np.trace(np.dot(W_true,W_true.T) + S) 

        shvar = 1
        spvar = 7.5
        var_path = f'{exp_dir}/variances_true{i+1}.xlsx'
        relvar_path = f'{exp_dir}/relative_variances_true{i+1}.xlsx'
        ind1, ind2 = compute_variances(W_true, res[i].d,total_var, shvar, spvar, var_path,relvar_path)

        print('True components of view 1: ',ind1, file=ofile)
        print('True components of view 2: ',ind2, file=ofile)      
        
        #plot estimated projections
        W1 = res[i].means_w[0]
        W2 = res[i].means_w[1]
        W = np.concatenate((W1, W2), axis=0)
        if W1.shape[1] == W_true.shape[1]:
            #match components
            W, comp_e, flip = match_comps(W, W_true)      
        if 'lowD' in filepath:                  
            W_path = f'{exp_dir}/estimated_W_{i+1}{file_ext}'      
            fig = plt.figure()
            plot_weights(W, W_path)

        #Compute estimated variances       
        if 'PCA' in filepath:
            S1 = 1/res[i].E_tau[0] * np.ones((1, W1.shape[0]))[0]
            S2 = 1/res[i].E_tau[1] * np.ones((1, W2.shape[0]))[0]
            S = np.diag(np.concatenate((S1, S2), axis=0))
        else:
            S1 = 1/res[i].E_tau[0]
            S2 = 1/res[i].E_tau[1]
            S = np.diag(np.concatenate((S1, S2), axis=1)[0,:])
        total_var = np.trace(np.dot(W,W.T) + S) 

        shvar = 1
        spvar = 7.5
        var_path = f'{exp_dir}/variances_est{i+1}.xlsx'
        relvar_path = f'{exp_dir}/relative_variances_est{i+1}.xlsx'
        ind1, ind2 = compute_variances(W, res[i].d,total_var, shvar, spvar, var_path,relvar_path)

        print('Estimated components of view 1: ',ind1, file=ofile)
        print('Estimated components of view 2: ',ind2, file=ofile)   

        # plot estimated latent variables
        Z_path = f'{exp_dir}/estimated_Z_{i+1}{file_ext}'
        x = np.linspace(0, res[i].means_z.shape[0], res[i].means_z.shape[0])
        numsub = res[i].means_z.shape[1]
        fig = plt.figure(figsize=(4.5, 5))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for j in range(numsub):
            ax = fig.add_subplot(numsub, 1, j+1)
            if W1.shape[1] == W_true.shape[1]:
                ax.scatter(x, res[i].means_z[:, comp_e[j]] * flip[j])
            else:
                ax.scatter(x, res[i].means_z[:, j])
        plt.savefig(Z_path)
        plt.close()

        # plot true latent variables
        Z_path = f'{exp_dir}/true_Z_{i+1}{file_ext}'
        x = np.linspace(0, res[i].Z.shape[0], res[i].Z.shape[0])
        numsub = res[i].Z.shape[1]
        fig = plt.figure(figsize=(4.5, 5))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for j in range(1, numsub+1):
            ax = fig.add_subplot(numsub, 1, j)
            ax.scatter(x, res[i].Z[:, j-1])
        plt.savefig(Z_path)
        plt.close()

        #plot estimated alphas
        a_path = f'{exp_dir}/estimated_alphas_{i+1}{file_ext}'
        color = 'white'
        a1 = np.reshape(res[i].E_alpha[0], (res[i].k, 1))
        a2 = np.reshape(res[i].E_alpha[1], (res[i].k, 1))
        a = np.concatenate((a1, a2), axis=1)
        fig = plt.figure(figsize=(2, 1.5))
        if W1.shape[1] == W_true.shape[1]:
            hinton(-a[comp_e,:].T, a_path, color) 
        else:
            hinton(-a.T, a_path, color)         

        #plot true alphas
        a_path = f'{exp_dir}/true_alphas_{i+1}{file_ext}'
        a1 = np.reshape(res[i].alphas[0], (res[i].alphas[0].shape[0], 1))
        a2 = np.reshape(res[i].alphas[1], (res[i].alphas[1].shape[0], 1))
        a = np.concatenate((a1, a2), axis=1)
        fig = plt.figure(figsize=(2, 1.5))
        hinton(-a.T, a_path, color)        

        # plot lower bound
        L_path = f'{exp_dir}/LB_{i+1}{file_ext}'
        fig = plt.figure(figsize=(5, 4))
        plt.plot(res[i].L[1:])
        plt.savefig(L_path)
        plt.close()  

    if 'training' in filepath and 'missing' in filepath:    

        #plot estimated projections
        W1 = res[best_init-1].W[0]
        W2 = res[best_init-1].W[1]
        W_true = np.concatenate((W1, W2), axis=0)
        W1 = res1[best_init-1].means_w[0]
        W2 = res1[best_init-1].means_w[1]
        W = np.concatenate((W1, W2), axis=0)
        if W1.shape[1] == W_true.shape[1]:
            #match components
            W, comp_e, flip = match_comps(W, W_true)                  
        W_path = f'{exp_dir}/estimated_W_MODEL2{file_ext}'      
        fig = plt.figure()
        color = 'grey'
        hinton(W, W_path, color)

        # plot estimated latent variables
        Z_path = f'{exp_dir}/estimated_Z_MODEL2{file_ext}'
        x = np.linspace(0, res1[best_init-1].means_z.shape[0], res1[best_init-1].means_z.shape[0])
        numsub = res1[best_init-1].means_z.shape[1]
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for j in range(numsub):
            ax = fig.add_subplot(numsub, 1, j+1)
            if W1.shape[1] == W_true.shape[1]:
                ax.scatter(x, res1[best_init-1].means_z[:, comp_e[j]] * flip[j])
            else:
                ax.scatter(x, res1[best_init-1].means_z[:, j])
        plt.savefig(Z_path)
        plt.close()   

        #plot estimated projections
        W1 = res2[best_init-1].means_w[0]
        W2 = res2[best_init-1].means_w[1]
        W = np.concatenate((W1, W2), axis=0)
        if W1.shape[1] == W_true.shape[1]:
            #match components
            W, comp_e, flip = match_comps(W, W_true)                   
        W_path = f'{exp_dir}/estimated_W_MODEL3{file_ext}'      
        fig = plt.figure()
        hinton(W, W_path, color)

        # plot estimated latent variables
        Z_path = f'{exp_dir}/estimated_Z_MODEL3{file_ext}'
        x = np.linspace(0, res2[best_init-1].means_z.shape[0], res2[best_init-1].means_z.shape[0])
        numsub = res2[best_init-1].means_z.shape[1]
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for j in range(numsub):
            ax = fig.add_subplot(numsub, 1, j+1)
            if W1.shape[1] == W_true.shape[1]:
                ax.scatter(x, res2[best_init-1].means_z[:, comp_e[j]] * flip[j])
            else:
                ax.scatter(x, res2[best_init-1].means_z[:, j])
        plt.savefig(Z_path)
        plt.close()

    #Saving file
    best_init = int(np.argmax(Lower_bounds)+1)
    print('\nOverall results--------------------------', file=ofile)
    print('Lower bounds: ', Lower_bounds[0], file=ofile)    
    print('Best initialisation: ', best_init, file=ofile)

    if 'missing' in filepath:
        for i in range(len(res[0].vmiss)):
            print(f'Mean rMSE (missing data in view {res[0].vmiss[i]}): ', np.mean(MSE_miss[i,:]), file=ofile)
            print(f'Std rMSE (missing data in view {res[0].vmiss[i]}): ', np.std(MSE_miss[i,:]), file=ofile)  

    ofile.close()          

                    
        

        
   