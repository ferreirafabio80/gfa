import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import xlsxwriter
import plotly.graph_objects as go
from scipy import io

def hinton(matrix, path, max_weight=None, ax=None):

    # Draw Hinton diagram for visualizing a weight matrix.
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
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

def plot_wbrain(comp, w_brain, l_brain, path_brain):
    #Brain weights
    plt.figure(figsize=(15, 10))
    #remove zero weights
    weight = w_brain
    w = weight[weight != 0]
    labels = l_brain[weight != 0]
    #sort weights and find top 40
    w_ind = np.argsort(-w)
    if w_ind.shape[0] > 50:
        top = 20
        w_top = np.concatenate((w_ind[0:top], w_ind[w_ind.shape[0]-top:w_ind.shape[0]]))
        w_sort = w[w_top]
        l_sort = labels[w_top]
    else:
        w_sort = w[w_ind]
        l_sort = labels[w_ind]    
    #color bars given sign of the weights
    colors = []     
    for k in range(0,w_sort.shape[0]): 
        if w_sort[k] > 0:
            colors.append('r')
        else:
            colors.append('b')
    x = range(w_sort.shape[0])        
    plt.barh(l_sort[::-1], width=w_sort[::-1], height=0.75,color=colors[::-1])
    plt.subplots_adjust(left=0.25,right=0.95)
    plt.title(f'Brain weights - Component {comp+1}',fontsize=18)
    plt.xlim([-1, 1])
    plt.yticks(x, l_sort[::-1], size='small')    
    plt.savefig(path_brain)
    plt.close()

def plot_wcli_mmse(var, w, label, categ):
    comp = w.shape[1]   
    for j in range(0, comp):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        #color bars given categories    
        #x = range(w.shape[0])
        cmap = cm.get_cmap('nipy_spectral')
        for k in range(0,w.shape[0]): 
            if categ[k] == 'Att. & Calc':
                bar1 = ax.bar(label[k], w[k,j], color=cmap(0.5))
            elif categ[k] == 'Demographics':
                bar2 = ax.bar(label[k], w[k,j], color=cmap(140))    
            elif categ[k] == 'Language':
                bar3 = ax.bar(label[k], w[k,j], color=cmap(35))
            elif categ[k] == 'Orientation':
                bar4 = ax.bar(label[k], w[k,j], color=cmap(100))               
            elif categ[k] == 'Recall':
                bar5 = ax.bar(label[k], w[k,j], color=cmap(175))
            elif categ[k] == 'Registration':
                bar6 = ax.bar(label[k], w[k,j], color=cmap(240))    
       
        filepath = f'{directory}/w_cli{j+1}.png'
        plt.subplots_adjust(left=0.25,right=0.95)
        plt.title(f'Clinical weights - Component {j+1}',fontsize=18)
        ax.legend((bar1[0], bar2[0],bar3[0],bar4[0],bar5[0],bar6[0]), (np.unique(categ)), title="Category")
        plt.ylim([-0.8, 0.8])  
        plt.savefig(filepath)
        plt.close()                   
    
def plot_Z(comp, values, ptype, path_z):
    fig = plt.figure(figsize=(25, 20))
    #fig = plt.figure(figsize=(20, 18))
    fig.subplots_adjust(hspace=0.75, wspace=0.75)
    plt.rc('font', size=10)
    numcomp = comp.shape[1]
    marker_size = 10
    for i in range(numcomp):
        index = numcomp*i + i + 1
        comp2 = 0
        y_comp = numcomp - i
        for j in range(y_comp):
            comp2 = i + j             
            ax = fig.add_subplot(numcomp, numcomp, index)
            if ptype == 'diagnosis':
                for N,k in zip(range(comp.shape[0]),values): 
                    if k == 1:
                        ax.scatter(comp[N,i], comp[N,comp2], c='green', s = marker_size, alpha=0.6, edgecolors='none')
                    elif k == 2:
                        ax.scatter(comp[N,i], comp[N,comp2], c='red', s = marker_size, alpha=0.6, edgecolors='none')
                    else:
                        ax.scatter(comp[N,i], comp[N,comp2], c='orange', s = marker_size, alpha=0.6, edgecolors='none')
            elif ptype == 'gender':
                for N,k in zip(range(comp.shape[0]),values): 
                    if k == 0:
                        ax.scatter(comp[N,i], comp[N,comp2], c='red', s = marker_size, alpha=0.6, edgecolors='none')
                    else:
                        ax.scatter(comp[N,i], comp[N,comp2], c='blue', s = marker_size, alpha=0.6, edgecolors='none')
            elif ptype == 'age':
                cmap = cm.get_cmap('copper')
                normalize = mpl.colors.Normalize(vmin=min(values), vmax=max(values))
                colors = [cmap(normalize(value)) for value in values]
                ax.scatter(comp[:,i], comp[:,comp2], c=colors, s = marker_size, alpha=0.6, edgecolors='none')
                #Add colorbar
                cax, _ = mpl.colorbar.make_axes(ax)
                mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
            ax.set_ylabel(f'Component {comp2+1}',fontsize=12)
            ax.set_xlabel(f'Component {i+1}', fontsize=12)
            ax.set_xlim(np.min(comp[:,i]),np.max(comp[:,i]))
            ax.set_ylim(np.min(comp[:,comp2]),np.max(comp[:,comp2]))  
            index += numcomp  
    plt.savefig(Z_path)
    plt.close()

def plot_wcli(var, w_cli, l_cli, path_cli):
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Clinical weights',fontsize=20)
    fig.subplots_adjust(hspace=1.3, wspace=0.5)
    comp = w_cli.shape[1]
    for j in range(0, comp):
        ax = fig.add_subplot(comp, 1, j+1)
        ax.set_title(f'Component {j+1}: {var[j]}',size=10)
        ax.set_ylim([-1, 1])
        
        #sort weights and find top 40
        w = w_cli[:, j]
        if w.shape[0] > 50:
            top = 20
            w_top = np.concatenate((w[0:top], w[w.shape[0]-top:w.shape[0]]))
            w = w[w_top]
            l_cli = l_cli[w_top]

        #color bars given sign of the weights
        colors = []
        for k in range(0,w.shape[0]): 
            if w[k] > 0:
                colors.append('r')
            else:
                colors.append('b')
        ax.bar(l_cli, w, color=colors)
    plt.savefig(path_cli)
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

#Settings
proj_dir = 'results'#'/cs/research/medic/human-connectome/experiments/fabio_hcp500'
data = 'simulations_lowD'
flag = ''
missing = False
if missing is True:
    p_miss = 40
    remove = 'rows'
    scenario = f'missing{str(p_miss)}_{remove}_view2'
else:
    scenario = 'complete'
model = 'GFA'
noise = 'PCA'
m = 15

#directories
directory = f'{proj_dir}/{data}/{flag}/{noise}/{m}models/{scenario}/'        
filepath = f'{directory}{model}_results.dictionary'

#Load file
with open(filepath, 'rb') as parameters:
    res = pickle.load(parameters) 

if 'simulations' not in data:
    for i in range(0, len(res)): #len(res)
        #Weights and total variance
        W1 = res[i].means_w[0]
        W2 = res[i].means_w[1]
        W = np.concatenate((W1, W2), axis=0)        
        if 'highD' in data:
            S1 = res[i].E_tau[0] * np.ones((1, W1.shape[0]))[0]
            S2 = res[i].E_tau[1] * np.ones((1, W2.shape[0]))[0]
            total_var = np.trace(np.dot(W1,W1.T) + S1) + np.trace(np.dot(W2,W2.T) + S2)                
        else:
            if 'PCA' in noise:
                S1 = res[i].E_tau[0] * np.ones((1, W1.shape[0]))[0]
                S2 = res[i].E_tau[1] * np.ones((1, W2.shape[0]))[0]
                S = np.diag(np.concatenate((S1, S2), axis=0))
                #S = np.concatenate((S1, S2), axis=0)
            elif 'FA' in noise:
                S = np.diag(np.concatenate((res[i].E_tau[0], res[i].E_tau[1]), axis=1)[0,:])
            total_var = np.trace(np.dot(W,W.T) + S) 
            #total_var = np.sum(W ** 2) + res[i].E_tau[0] * W1.shape[0] + res[i].E_tau[1] * W2.shape[0]     
        #Explained variance
        var1 = np.zeros((1, W1.shape[1]))
        var2 = np.zeros((1, W2.shape[1]))
        var = np.zeros((1, W.shape[1]))
        for c in range(0, W.shape[1]):
            w = np.reshape(W[:,c],(W.shape[0],1))
            w1 = np.reshape(W1[:,c],(W1.shape[0],1))
            w2 = np.reshape(W2[:,c],(W2.shape[0],1))
            var1[0,c] = (np.trace(np.dot(w1.T, w1))/total_var) * 100
            var2[0,c] = (np.trace(np.dot(w2.T, w2))/total_var) * 100
            var[0,c] = (np.trace(np.dot(w.T, w))/total_var) * 100

        """ var_path = f'{directory}/variances{i+1}.xlsx'
        df = pd.DataFrame({'components':range(1, W.shape[1]+1),'Brain': list(var1[0,:]),'Behaviour': list(var2[0,:]), 'Both': list(var[0,:])})
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(var_path, engine='xlsxwriter')

        # Convert the dataframe to an XlsxWriter Excel object.
        df.to_excel(writer, sheet_name='Sheet1')

        # Close the Pandas Excel writer and output the Excel file.
        writer.save() """

        #ind = np.array((3,7,8,9,10,22,23))      
        ind = np.array((2,4,6,8,15,17,19))      
        #sort components
        """ ind1 = np.argsort(var1)
        ind2 = np.argsort(var2)
        var_sorted = np.sort(var)        
        if 'highD' in data:
            #components explaining >1% variance
            ind = np.flip(ind[var_sorted >= 1])
        else:
            ind = np.flip(ind[var_sorted >= 0.01])
        numcomp = ind.shape[0] """    

        if 'ADNI' in data:
            data_dir = f'results/{data}/data'
            #Labels
            if 'highD' in data:
                clinical_labels = pd.read_csv(f'{data_dir}/Y_labels.csv')
                groups = pd.read_csv(f'{data_dir}/groups.csv')
                Y_labels = clinical_labels.Labels.values
                Y_categ = clinical_labels.Categories.values
                #Clinical weights - MMSE
                plot_wcli_mmse(var, W2[:,ind], Y_labels, Y_categ)
                brain_weights = {"wx": W1[:,ind]}
                io.savemat(f'{directory}/wx.mat', brain_weights)
            elif 'lowD' in data:
                brain_labels = pd.read_csv(f'{data_dir}/X_labels_clean.csv')
                clinical_labels = pd.read_csv(f'{data_dir}/Y_labels_splitgender.csv')
                groups = pd.read_csv(f'{data_dir}/groups.csv')
                X_labels = brain_labels.Regions.values
                Y_labels = clinical_labels.clinical.values 
                #Brain weights
                for j in range(0, numcomp):  
                    brain_path = f'{directory}/w_brain{i+1}_comp{j+1}.png'
                    plot_wbrain(j, W1[:,ind[j]], X_labels, brain_path)

                #Clinical weights
                cli_path = f'{directory}/w_cli{i+1}.png'
                plot_wcli(var[0,ind], W2[:,ind], Y_labels, cli_path)
            cohort = groups.cohort.values
            gender = groups.gender.values
            age = groups.age.values
        
        else:           
            #Clinical weights
            brain_weights = {"wx": W1[:,ind]}
            io.savemat(f'{directory}/wx{i+1}.mat', brain_weights)
            #Brain weights
            clinical_weights = {"wy": W2[:,ind]}
            io.savemat(f'{directory}/wy{i+1}.mat', clinical_weights)

            #group info
            """ data_dir = f'results/{data}/{flag}/data'
            groups = pd.read_csv(f'{data_dir}/groups.csv')
            if 'NSPN' in data:
                cohort = groups.cohort.values
                gender = groups.gender.values
                age = groups.age.values
            elif 'ABCD' in data:
                gender = groups.gender.values """      
        
        """ #Latent spaces
        #-----------------------------------------------------------
        comps = res[i].means_z[:,ind]
        #Colored by age
        if 'age' in locals():
            plottype = 'age'
            Z_path = f'{directory}/LS_{plottype}{i+1}.svg'
            plot_Z(comps, age, plottype, Z_path)
        if 'cohort' in locals():    
            #Colored by diagnosis
            plottype = 'diagnosis'
            Z_path = f'{directory}/LS_{plottype}{i+1}.svg'
            plot_Z(comps, cohort, plottype, Z_path)
        if 'gender' in locals():
            #Colored by gender
            plottype = 'gender'
            Z_path = f'{directory}/LS_{plottype}{i+1}.svg'
            plot_Z(comps, gender, plottype, Z_path)  """

        #Plot lower bound
        L_path = f'{directory}/LB{i+1}.png'
        plt.figure()
        plt.title('Lower Bound')
        plt.plot(res[i].L[1:])
        plt.savefig(L_path)
        plt.close()

else:
    if 'missing' in scenario:
        file_missing = f'{directory}{model}_results_imputation.dictionary'
        with open(file_missing, 'rb') as parameters:
            res1 = pickle.load(parameters)

    for i in range(0, len(res)):

        #plot predictions
        obs_view = np.array([1, 0])
        #view 2 from view 1
        vpred1 = np.where(obs_view == 0)
        if 'missing' in scenario:
            df = pd.DataFrame(columns=['x', 'Pred_nomissing','Pred_imputation','Pred_mean'])
            for j in range(res[i].d[vpred1[0][0]]):
                df = df.append({'x':j+1, 'Pred_nomissing': res[i].reMSE1[0,j], 
                'Pred_imputation': res1[i].reMSE1[0,j], 'Pred_mean': res[i].reMSEmean1[0,j]}, ignore_index=True)
            ymax = max(np.max(res[i].reMSE1),np.max(res1[i].reMSE1), np.max(res[i].reMSEmean1))
            title = f'Predict view 2 from view 1 ({str(p_miss)}% missing {remove})'    
        else:
            df = pd.DataFrame(columns=['x', 'Pred_nomissing','Pred_mean'])
            for j in range(res[i].d[vpred1[0][0]]):
                df = df.append({'x':j+1, 'Pred_nomissing': res[i].reMSE1[0,j], 
                    'Pred_mean': res[i].reMSEmean1[0,j]}, ignore_index=True)
            ymax = max(np.max(res[i].reMSE1), np.max(res[i].reMSEmean1))         
            title = f'Predict view 2 from view 1 (complete)'
        line_path = f'{directory}/predictions_view2_{i+1}.png'         
        plot_predictions(df, ymax, title, line_path)

        #view 1 from view 2
        vpred2 = np.where(obs_view == 1)
        if 'missing' in scenario:
            df = pd.DataFrame(columns=['x', 'Pred_nomissing','Pred_imputation','Pred_mean'])
            for j in range(res[i].d[vpred2[0][0]]):
                df = df.append({'x':j+1, 'Pred_nomissing': res[i].reMSE2[0,j], 
                'Pred_imputation': res1[i].reMSE2[0,j], 'Pred_mean': res[i].reMSEmean2[0,j]}, ignore_index=True)
            title = f'Predict view 1 from view 2 ({str(p_miss)}% missing {remove})'
            ymax = max(np.max(res[i].reMSE2),np.max(res1[i].reMSE2), np.max(res[i].reMSEmean2))

        else:
            df = pd.DataFrame(columns=['x', 'Pred_nomissing','Pred_mean'])
            for j in range(res[i].d[vpred2[0][0]]):
                df = df.append({'x':j+1, 'Pred_nomissing': res[i].reMSE2[0,j], 
                    'Pred_mean': res[i].reMSEmean2[0,j]}, ignore_index=True)
            title = f'Predict view 1 from view 2 (complete)'
            ymax = max(np.max(res[i].reMSE2), np.max(res[i].reMSEmean2))                 
        line_path = f'{directory}/predictions_view1_{i+1}.png'
        plot_predictions(df, ymax, title, line_path)  

        #Tables
        if missing is True:
            table_path = f"{directory}/table_{i+1}.png"
            fig = go.Figure(data=[go.Table(
                header=dict(values=['<b>Views<b>', '<b>True Prediction</b><br>(Frobenius norm)'
                    , '<b>Prediction with imputation</b><br>(Frobenius norm)', '<b>Prediction Mean</b><br>(Frobenius norm)'],
                            fill_color='paleturquoise',
                            align='center'),
                cells=dict(values=[[1, 2], # 1st column
                                [res[i].Fnorm2,res[i].Fnorm1],
                                [res1[i].Fnorm2,res1[i].Fnorm1],
                                [res[i].Fnorm_mean2,res[i].Fnorm_mean1]], # 2nd column
                        fill_color='lavender',
                        align='center'))
            ])

            fig.update_layout(width=1000, height=500)
            fig.write_image(table_path)
        else:
            table_path = f"{directory}/table_{i+1}.png"
            fig = go.Figure(data=[go.Table(
                header=dict(values=['<b>Views<b>', '<b>True Prediction</b><br>(Frobenius norm)', '<b>Prediction Mean</b><br>(Frobenius norm)'],
                            fill_color='paleturquoise',
                            align='center'),
                cells=dict(values=[[1, 2], # 1st column
                                [res[i].Fnorm2,res[i].Fnorm1],
                                [res[i].Fnorm_mean2,res[i].Fnorm_mean1]], # 2nd column
                        fill_color='lavender',
                        align='center'))
            ])

            #fig.update_layout(width=500, height=300)
            fig.write_image(table_path)    

        # Hinton diagrams for W1 and W2
        W1 = res[i].means_w[0]
        W2 = res[i].means_w[1]
        W = np.concatenate((W1, W2), axis=0)
        S1 = res[i].E_tau[0] * np.ones((1, W1.shape[0]))[0]
        S2 = res[i].E_tau[1] * np.ones((1, W2.shape[0]))[0]
        total_var = np.trace(np.dot(W1,W1.T) + S1) + np.trace(np.dot(W2,W2.T) + S2)
        #Explained variance
        var = np.zeros((1, W.shape[1]))
        for c in range(0, W.shape[1]):
            w = np.reshape(W[:,c],(W.shape[0],1))
            var[0,c] = (np.trace(np.dot(w.T, w))/total_var) * 100

        #sort components
        ind = np.argsort(var)
        var_sorted = np.sort(var)        
        ind = np.flip(ind[var_sorted >= 0.4])
        W_path = f'{directory}/estimated_Ws{i+1}.png'
        fig = plt.figure()
        fig.suptitle('Estimated Ws')
        hinton(W[:,ind], W_path)

        # plot estimated latent variables
        Z_path = f'{directory}/estimated_Z{i+1}.png'
        x = np.linspace(0, res[i].means_z.shape[0], res[i].means_z.shape[0])
        numsub = res[i].means_z.shape[1]
        fig = plt.figure()
        fig.suptitle('Estimated latent components')
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for j in range(1, numsub+1):
            ax = fig.add_subplot(numsub, 1, j)
            ax.scatter(x, res[i].means_z[:, ind[j-1]])
        plt.savefig(Z_path)
        plt.close()

        # Hinton diagrams for alpha1 and alpha2
        a_path = f'{directory}/estimated_alphas{i+1}.png'
        a1 = np.reshape(res[i].E_alpha[0], (res[i].m, 1))
        a2 = np.reshape(res[i].E_alpha[1], (res[i].m, 1))
        a = np.concatenate((a1, a2), axis=1)
        fig = plt.figure()
        fig.suptitle('Estimated Alphas')
        hinton(-a[ind,:].T, a_path)

        # plot lower bound
        L_path = f'{directory}/LB{i+1}.png'
        fig = plt.figure()
        fig.suptitle('Lower Bound')
        plt.plot(res[i].L[1:])
        plt.savefig(L_path)
        plt.close()

        # plot true projections
        W_path = f'{directory}/true_Ws{i+1}.png'
        W1 = res[i].W[0]
        W2 = res[i].W[1]
        W = np.concatenate((W1, W2), axis=0)
        fig = plt.figure()
        fig.suptitle('True Ws')
        hinton(W, W_path)
        plt.close()

        # plot true latent variables
        Z_path = f'{directory}/true_Z{i+1}.png'
        x = np.linspace(0, res[i].Z.shape[0], res[i].Z.shape[0])
        numsub = res[i].Z.shape[1]
        fig = plt.figure()
        fig.suptitle('True latent components')
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for j in range(1, numsub+1):
            ax = fig.add_subplot(numsub, 1, j)
            ax.scatter(x, res[i].Z[:, j-1])
        plt.savefig(Z_path)
        plt.close()

        

        
    

        
   