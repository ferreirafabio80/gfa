import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import xlsxwriter
import copy
from scipy import io
from utils import GFAtools
from sklearn.metrics.pairwise import cosine_similarity

def hinton(matrix, path, max_weight=None, ax=None):
    # Draw Hinton diagram for visualizing a weight matrix.
    plt.figure(figsize=(2, 1.5))
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
    plt.figure(figsize=(2.5, 2))
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

def results_HCP(ninit, beh_dim, exp_dir):

    #Output file
    ofile = open(f'{exp_dir}/output.txt','w')

    if 'training' in exp_dir:
        MSE = np.zeros((1,ninit))
        MSE_beh = np.zeros((ninit, beh_dim))
    if 'missing' in exp_dir:
        MSEmissing = np.zeros((1,ninit))    
    LB = np.zeros((1,ninit))
    file_ext = '.png'
    for i in range(ninit):
        
        print('\nInitialisation: ', i+1, file=ofile)
        print('------------------------------------------------', file=ofile)
        filepath = f'{exp_dir}GFA_results{i+1}.dictionary'
        #Load file
        with open(filepath, 'rb') as parameters:
            res = pickle.load(parameters)  

        #Save LB values
        LB[0,i] = res.L[-1]       

        #checking NaNs
        if 'missing' in filepath:
            if 'view1' in filepath:
                total = res.X_nan[0].size
                n_miss = np.flatnonzero(res.X_nan[0]).shape[0]
                print(f'Percentage of missing data in view 1: {round((n_miss/total)*100)}', file=ofile)
            else:
                total = res.X_nan[1].size
                n_miss = np.flatnonzero(res.X_nan[1]).shape[0]
                print(f'Percentage of missing data in view 2: {round((n_miss/total)*100)}', file=ofile)

        if 'training' in filepath:
            N_train = res.N
            N_test = res.X_test[0].shape[0]
            print('Percentage of train data: ', round(N_train/(N_test+N_train)*100), file=ofile)

        #Computational time
        print('Computational time (hours): ', round(res.time_elapsed/3600), file=ofile)

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
        """ ind1 = []
        ind2 = []  
        for k in range(W.shape[1]):
            if res.E_alpha[0][k] < 300:
                ind1.append(k)    
            if res.E_alpha[1][k] < 300:
                ind2.append(k) """
                
        shvar = 2
        spvar = 15
        var_path = f'{exp_dir}/variances{i+1}.xlsx'
        relvar_path = f'{exp_dir}/relative_variances{i+1}.xlsx'
        ind1, ind2 = compute_variances(W, res.d,total_var, shvar, spvar, var_path,relvar_path)                   
        
        #Components
        print('Brain components: ', ind1, file=ofile)
        print('Clinical components: ', ind2, file=ofile)

        #Clinical weights
        if len(ind1) > 0:
            brain_weights = {"wx": W1[:,np.array(ind1)]}
            io.savemat(f'{exp_dir}/wx{i+1}.mat', brain_weights)
        #Brain weights
        if len(ind2) > 0:
            clinical_weights = {"wy": W2[:,np.array(ind2)]}
            io.savemat(f'{exp_dir}/wy{i+1}.mat', clinical_weights)

        #Plot lower bound
        L_path = f'{exp_dir}/LB{i+1}{file_ext}'
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
                    v_miss = 0
                elif 'view2' in filepath:
                    obs_view = np.array([1, 0])
                    v_miss = 1
                mask_miss = res.X_nan[v_miss]==1
                X = copy.deepcopy(res.X_train)        
                missing_true = np.where(mask_miss,X[v_miss],0)       
                X[v_miss][mask_miss] = 'NaN'
                #predict missing values
                missing_pred = GFAtools(X, res, obs_view).PredictMissing()
                miss_true = missing_true[mask_miss]
                miss_pred = missing_pred[v_miss][mask_miss]
                MSEmissing[0,i] = np.mean((miss_true - miss_pred) ** 2)
                print('MSE for missing data: ', MSEmissing[0,i], file=ofile)
        
            obs_view = np.array([1, 0])
            vpred = np.array(np.where(obs_view == 0))
            X_pred = GFAtools(res.X_test, res, obs_view).PredictView(noise)

            #-Metrics
            #----------------------------------------------------------------------------------
            MSE[0,i] = np.mean((res.X_test[vpred[0,0]] - X_pred) ** 2)
            #MSE for each dimension - predict view 2 from view 1
            for j in range(0, beh_dim):
                MSE_beh[i,j] = np.mean((res.X_test[vpred[0,0]][:,j] - X_pred[:,j]) ** 2)/np.mean(res.X_test[vpred[0,0]][:,j] ** 2)
    
    best_init = int(np.argmax(LB)+1)
    print('\nOverall results--------------------------', file=ofile)
    print('Lower bounds: ', LB[0], file=ofile)    
    print('Best initialisation: ', best_init, file=ofile)
    if 'missing' in filepath:
        print(f'Avg. MSE (missing data): ', np.mean(MSEmissing), file=ofile)
        print(f'Std MSE(missing data): ', np.std(MSEmissing), file=ofile)  

    if 'training' in filepath:
        print(f'Avg. MSE: ', np.mean(MSE), file=ofile)
        print(f'Std MSE: ', np.std(MSE), file=ofile) 
        #Predictions for behaviour
        #---------------------------------------------
        plt.figure(figsize=(10,8))
        pred_path2 = f'{exp_dir}/Predictions_v2{file_ext}'
        x = np.arange(MSE_beh.shape[1])
        plt.errorbar(x, np.mean(MSE_beh,axis=0), yerr=np.std(MSE_beh,axis=0), fmt='o', label='Obs. data only')
        plt.legend(loc='upper right',fontsize=14)
        plt.ylim((0,1.5))
        plt.xlabel('Features of view 2',fontsize=16)
        plt.ylabel('RMSE',fontsize=16)
        plt.savefig(pred_path2)
        plt.close()  
     
    ofile.close()     

def results_simulations(exp_dir):
    
    #Load file
    filepath = f'{exp_dir}/GFA_results.dictionary'
    with open(filepath, 'rb') as parameters:
        res = pickle.load(parameters)

    #Output file
    ofile = open(f'{exp_dir}/output.txt','w')    
    
    if 'missing' in filepath:
        MSE_miss = np.zeros((len(res[0].vmiss), len(res)))

    if 'training' in filepath:
        MSE_v1 = np.zeros((1, len(res)))
        MSE_v2 = np.zeros((1, len(res)))
        if 'missing' in filepath:
            MSEimp_v1 = np.zeros((1, len(res)))
            MSEimp_v2 = np.zeros((1, len(res)))
            MSEmed_v1 = np.zeros((1, len(res)))
            MSEmed_v2 = np.zeros((1, len(res)))
            LB_imp = np.zeros((1,len(res))) 
            LB_med = np.zeros((1,len(res)))         

    LB = np.zeros((1,len(res)))    
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
                print(f'Percentage of missing data in view {v_miss}: {round((n_miss/total)*100)}', file=ofile)

                print(f'MSE missing data (view {res[i].vmiss[j]}):', res[i].MSEmissing[j], file=ofile)
                MSE_miss[j,i] = res[i].MSEmissing[j]
        
        #Save LB values
        LB[0,i] = res[i].L[-1] 

        if 'training' in filepath:
            N_train = res[i].N
            N_test = res[i].N_test
            print('Percentage of train data: ', round(N_train/(N_test+N_train)*100), file=ofile)
            
            if 'missing' in filepath:
                file_missing = f'{exp_dir}/GFA_results_imputation.dictionary'
                with open(file_missing, 'rb') as parameters:
                    res1 = pickle.load(parameters)

                file_median = f'{exp_dir}/GFA_results_median.dictionary'
                with open(file_median, 'rb') as parameters:
                    res2 = pickle.load(parameters)

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

        # plot true Ws
        W1 = res[i].W[0]
        W2 = res[i].W[1]
        W_true = np.concatenate((W1, W2), axis=0)
        W_path = f'{exp_dir}/W_true{i+1}{file_ext}'
        plot_weights(W_true, res[i].d, W_path)            

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
        
        #plot estimated Ws
        W1 = res[i].means_w[0]
        W2 = res[i].means_w[1]
        W = np.concatenate((W1, W2), axis=0)
        if W.shape[1] == W_true.shape[1]:
            #match components
            W, comp_e, flip = match_comps(W, W_true)                       
        W_path = f'{exp_dir}/W_est{i+1}{file_ext}'      
        plot_weights(W, res[i].d, W_path)

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
        Z_path = f'{exp_dir}/Z_est{i+1}{file_ext}'
        if W.shape[1] == W_true.shape[1]:
            plot_Z(res[i], comp_e, flip, Z_path)
        else:     
            plot_Z(res[i], path=Z_path, match=False)     

        # plot true latent variables
        Z_path = f'{exp_dir}/Z_true{i+1}{file_ext}'
        if W.shape[1] == W_true.shape[1]:
            plot_Z(res[i], comp_e, flip, Z_path)
        else:     
            plot_Z(res[i], path=Z_path, match=False)

        #plot estimated alphas
        a_path = f'{exp_dir}/alphas_est{i+1}{file_ext}'
        a1 = np.reshape(res[i].E_alpha[0], (res[i].k, 1))
        a2 = np.reshape(res[i].E_alpha[1], (res[i].k, 1))
        a = np.concatenate((a1, a2), axis=1)
        if W1.shape[1] == W_true.shape[1]:
            hinton(-a[comp_e,:].T, a_path) 
        else:
            hinton(-a.T, a_path)         

        #plot true alphas
        a_path = f'{exp_dir}/alphas_true{i+1}{file_ext}'
        a1 = np.reshape(res[i].alphas[0], (res[i].alphas[0].shape[0], 1))
        a2 = np.reshape(res[i].alphas[1], (res[i].alphas[1].shape[0], 1))
        a = np.concatenate((a1, a2), axis=1)
        hinton(-a.T, a_path)        

        # plot lower bound
        L_path = f'{exp_dir}/LB{i+1}{file_ext}'
        plt.figure(figsize=(5, 4))
        plt.plot(res[i].L[1:])
        plt.savefig(L_path)
        plt.close()

    #Overall results
    print('\nOverall results--------------------------', file=ofile)   
    print('Best initialisation: ', int(np.argmax(LB)+1), file=ofile)      

    if 'missing' in filepath:
        for i in range(len(res[0].vmiss)):
            print(f'Avg. MSE (missing data in view {res[0].vmiss[i]}): ', np.mean(MSE_miss[i,:]), file=ofile)
            print(f'Std MSE (missing data in view {res[0].vmiss[i]}): ', np.std(MSE_miss[i,:]), file=ofile)  
    
    if 'training' in filepath:
        #Predictions
        #View 1
        print('Predictions for view 1',file=ofile)
        print('Observed data----------------------------',file=ofile)
        print(f'Avg. MSE: ', np.mean(MSE_v1), file=ofile)
        print(f'Std MSE: ', np.std(MSE_v1), file=ofile) 
        if 'missing' in filepath: 
            print('Imputed data with predicted values--------',file=ofile)
            print(f'Avg. MSE: ', np.mean(MSEimp_v1), file=ofile)
            print(f'Std MSE: ', np.std(MSEimp_v1), file=ofile) 
            print('Imputed data with median------------------',file=ofile)
            print(f'Avg. MSE: ', np.mean(MSEmed_v1), file=ofile)
            print(f'Std MSE: ', np.std(MSEmed_v1), file=ofile)

        #View 2
        print('Predictions for view 2',file=ofile)
        print('Observed data----------------------------',file=ofile)
        print(f'Avg. MSE: ', np.mean(MSE_v2), file=ofile)
        print(f'Std MSE: ', np.std(MSE_v2), file=ofile)
        if 'missing' in filepath:   
            print('Imputed data with predicted values--------',file=ofile)
            print(f'Avg. MSE: ', np.mean(MSEimp_v2), file=ofile)
            print(f'Std MSE: ', np.std(MSEimp_v2), file=ofile) 
            print('Imputed data with median------------------',file=ofile)
            print(f'Avg. MSE: ', np.mean(MSEmed_v2), file=ofile)
            print(f'Std MSE: ', np.std(MSEmed_v2), file=ofile)          

        if 'missing' in filepath:  
            
            #MODEL IMPUTATION
            #-----------------------------------------------
            best_init = int(np.argmax(LB_imp)+1)
            W1 = res[best_init-1].W[0]
            W2 = res[best_init-1].W[1]
            W_true = np.concatenate((W1, W2), axis=0)
            #plot estimated projections            
            W1 = res1[best_init-1].means_w[0]
            W2 = res1[best_init-1].means_w[1]
            W = np.concatenate((W1, W2), axis=0)
            if W.shape[1] == W_true.shape[1]:
                #match components
                W, comp_e, flip = match_comps(W, W_true)                   
            W_path = f'{exp_dir}/W_est_IMPUTATION{file_ext}'      
            plot_weights(W, res[0].d, W_path)

            # plot estimated latent variables
            Z_path = f'{exp_dir}/Z_est_IMPUTATION{file_ext}'
            if W.shape[1] == W_true.shape[1]:
                plot_Z(res1[best_init-1], comp_e, flip, Z_path)
            else:     
                plot_Z(res1[best_init-1], path=Z_path, match=False)   

            #MODEL MEDIAN
            #-----------------------------------------------
            best_init = int(np.argmax(LB_med)+1)
            W1 = res[best_init-1].W[0]
            W2 = res[best_init-1].W[1]
            W_true = np.concatenate((W1, W2), axis=0)
            #plot estimated projections
            W1 = res2[best_init-1].means_w[0]
            W2 = res2[best_init-1].means_w[1]
            W = np.concatenate((W1, W2), axis=0)
            if W.shape[1] == W_true.shape[1]:
                #match components
                W, comp_e, flip = match_comps(W, W_true)                     
            W_path = f'{exp_dir}/W_est_MEDIAN{file_ext}'      
            plot_weights(W, res[0].d, W_path)

            # plot estimated latent variables
            Z_path = f'{exp_dir}/Z_est_MEDIAN{file_ext}'
            if W.shape[1] == W_true.shape[1]:
                plot_Z(res2[best_init-1], comp_e, flip, Z_path)
            else:     
                plot_Z(res2[best_init-1], path=Z_path, match=False)                     

    ofile.close()                 
        

        
   