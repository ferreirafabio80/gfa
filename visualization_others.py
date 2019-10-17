import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pickle
import pandas as pd
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

def plot_wcli_mmse(var, w_cli, l_cli):
    ind = np.argsort(var)
    var_sorted = np.sort(var)
    #components explaining >1% variance
    ind = np.flip(ind[var_sorted >= 1])
    comp = ind.shape[0]   
    for j in range(0, comp):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        #sort weights and find top 40
        w = w_cli[:, ind[j]]
       
        #color bars given categories    
        x = range(w.shape[0])
        cmap = cm.get_cmap('nipy_spectral')
        for k in range(0,w.shape[0]): 
            if l_cli[k] == 'Att. & Calc':
                bar1 = ax.bar(x[k], w[k], color=cmap(0.5))
            elif l_cli[k] == 'Language':
                bar2 = ax.bar(x[k], w[k], color=cmap(35))
            elif l_cli[k] == 'Orientation':
                bar3 = ax.bar(x[k], w[k], color=cmap(100))               
            elif l_cli[k] == 'Recall':
                bar4 = ax.bar(x[k], w[k], color=cmap(175))
            elif l_cli[k] == 'Registration':
                bar5 = ax.bar(x[k], w[k], color=cmap(240))
       
        filepath = f'{directory}/w_cli{j+1}.png'
        plt.subplots_adjust(left=0.25,right=0.95)
        plt.title(f'Clinical weights - Component {j+1}',fontsize=18)
        ax.legend((bar1[0], bar2[0],bar3[0],bar4[0],bar5[0]),(np.unique(l_cli)), title="Category")
        plt.ylim([-0.8, 0.8])  
        plt.savefig(filepath)
        plt.close()                   
    
def plot_Z(comp, values, ptype, path_z):
    #fig = plt.figure(figsize=(25, 20))
    fig = plt.figure(figsize=(15, 12))
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
                    if k == 'AD':
                        ax.scatter(comp[N,i], comp[N,comp2], c='red', s = marker_size, alpha=0.6, edgecolors='none')
                    elif k == 'CN':
                        ax.scatter(comp[N,i], comp[N,comp2], c='green', s = marker_size, alpha=0.6, edgecolors='none')
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
    ind = np.argsort(-var)
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Clinical weights',fontsize=20)
    fig.subplots_adjust(hspace=1.3, wspace=0.5)
    comp = w_cli.shape[1]
    for j in range(0, comp):
        ax = fig.add_subplot(comp, 1, j+1)
        ax.title.set_text(f'Component {j+1}')
        #ax.title.set_text(f'Component {j+1}')
        ax.set_ylim([-1, 1])
        #remove zero weights
        w = w_cli[:, ind[j]]
        #w = weight[weight != 0]
        #labels = l_cli[weight != 0]
        #sort weights and find top 40
        w_ind = np.argsort(-w)
        if w_ind.shape[0] > 50:
            top = 20
            w_top = np.concatenate((w_ind[0:top], w_ind[w_ind.shape[0]-top:w_ind.shape[0]]))
            w_sort = w[w_top]
            l_sort = l_cli[w_top]
        else:    
            #w_sort = w[w_ind]
            #l_sort = l_cli[w_ind]
            w_sort = w
            l_sort = l_cli
       
        #color bars given sign of the weights
        colors = []
        for k in range(0,w_sort.shape[0]): 
            if w_sort[k] > 0:
                colors.append('r')
            else:
                colors.append('b')
        ax.bar(l_sort, w_sort,color=colors)
    plt.savefig(path_cli)
    plt.close()

data = 'simulations'
flag = '_lowD'
machine = 'CCA'
directory = f'results/{data}{flag}/{machine}'   
data_dir = f'results/{data}/{flag}/data'

if 'ADNI' in data:
    #Labels
    if 'highD' in data:
        clinical_labels = pd.read_csv(f'{data_dir}/Y_labels.csv')
        groups = pd.read_csv(f'{data_dir}/groups.csv')
    elif 'lowD' in data:
        brain_labels = pd.read_csv(f'{data_dir}/X_labels_clean.csv')
        clinical_labels = pd.read_csv(f'{data_dir}/Y_labels.csv')
        groups = pd.read_csv(f'{data_dir}/groups.csv')
        X_labels = brain_labels.Regions.values
        Y_labels = clinical_labels.clinical.values 
    cohorts = groups.cohort.values
    gender = groups.gender.values
    age = groups.age.values
       
    if machine=='CCA':    
        fw = 'holdout' 
        #Clinical weights
        filepath = f'{directory}/wcli_{fw}.mat'
        cli_path = f'{directory}/wcli_{fw}.svg'
        weights = io.loadmat(filepath)
        W1 = weights['w2']
        var = np.array((30,20,10))
        plot_wcli(var, W1, Y_labels, cli_path)

        #Brain weights
        filepath = f'{directory}/wbrain_{fw}.mat'
        weights = io.loadmat(filepath)
        W2 = weights['w1']
        var = np.array((30,20,10))
        ind = np.argsort(-var)
        numcomp = W2.shape[1]
        for i in range(0, numcomp):  
            brain_path = f'{directory}/wbrain_{fw}{i+1}.png'
            plot_wbrain(i, W2[:,ind[i]], X_labels, brain_path)

        #Latent spaces
        X1 = io.loadmat(f'{directory}/X_clean.mat')
        X2 = io.loadmat(f'{directory}/Y_new.mat')
        comp1 = (np.dot(X1['X'],W2) + np.dot(X2['Y'],W1))/2
        comp2 = (np.dot(X1['X'],W2) + np.dot(X2['Y'],W1))/2
        comps = (comp1+comp2)/2
        #Colored by age
        plottype = 'age'
        Z_path = f'{directory}/LS_{plottype}.svg'
        plot_Z(comps, age, plottype, Z_path)
        #Colored by diagnosis
        plottype = 'diagnosis'
        Z_path = f'{directory}/LS_{plottype}.svg'
        plot_Z(comps, cohorts, plottype, Z_path)
        #Colored by gender
        plottype = 'gender'
        Z_path = f'{directory}/LS_{plottype}.svg'
        plot_Z(comps, gender, plottype, Z_path) 
           
    elif machine == 'SCCA':
        #Clinical weights
        filepath = f'{directory}/wcli.mat'
        cli_path = f'{directory}/wcli.svg'
        weights = io.loadmat(filepath)
        W1 = weights['v']
        var = np.array((30,20,10))
        plot_wcli(var, W1, Y_labels, cli_path)

        #Brain weights
        filepath = f'{directory}/wbrain.mat'
        weights = io.loadmat(filepath)
        W2 = weights['u']
        var = np.array((30,20,10))
        ind = np.argsort(-var)
        numcomp = W2.shape[1]
        for i in range(0, numcomp):  
            brain_path = f'{directory}/wbrain{i+1}.png'
            plot_wbrain(i, W2[:,ind[i]], X_labels, brain_path)

        #Latent spaces
        X1 = io.loadmat(f'{directory}/X_clean.mat')
        X2 = io.loadmat(f'{directory}/Y_new.mat')
        comp1 = (np.dot(X1['X'],W2) + np.dot(X2['Y'],W1))/2
        comp2 = (np.dot(X1['X'],W2) + np.dot(X2['Y'],W1))/2
        comps = (comp1+comp2)/2
        #Colored by age
        plottype = 'age'
        Z_path = f'{directory}/LS_{plottype}.svg'
        plot_Z(comps, age, plottype, Z_path)
        #Colored by diagnosis
        plottype = 'diagnosis'
        Z_path = f'{directory}/LS_{plottype}.svg'
        plot_Z(comps, cohorts, plottype, Z_path)
        #Colored by gender
        plottype = 'gender'
        Z_path = f'{directory}/LS_{plottype}.svg'
        plot_Z(comps, gender, plottype, Z_path)   

elif 'simulations' in data:
    #Latent spaces
    X1 = io.loadmat(f'{directory}/X1.mat')
    X2 = io.loadmat(f'{directory}/X2.mat')
    wx = io.loadmat(f'{directory}/wx.mat')
    wy = io.loadmat(f'{directory}/wy.mat')
    X = [[] for _ in range(2)]
    X[0] = X1['X1']
    X[1] = X2['X2']
    W = [[] for _ in range(2)]
    W[0] = wx['wx']
    W[1] = wy['wy']
    
    # Hinton diagrams for W1 and W2
    W_conc = np.concatenate((W[0], W[1]), axis=0)
    colMeans_W = np.mean(W_conc ** 2, axis=0)
    var = colMeans_W * 100
    ind = np.argsort(-var)
    W_path = f'{directory}/estimated_Ws.svg'
    fig = plt.figure()
    fig.suptitle('Estimated Ws')
    hinton(W_conc[:,ind], W_path)
    
    #Latent spaces
    Z_path = f'{directory}/estimated_Z.svg'
    x = np.linspace(0, X[0].shape[0]-1, X[0].shape[0])
    numsub = 2
    fig = plt.figure()
    fig.suptitle('Estimated latent components',fontsize=18)
    fig.subplots_adjust(hspace=0.7, wspace=0.4)
    for j in range(0, numsub):
        comp = (np.dot(X[0],W[0][:,j]) + np.dot(X[1],W[1][:,j]))/2
        ax = fig.add_subplot(numsub, 1, j+1)
        ax.scatter(x, -comp)
        ax.title.set_text(f'Component {j+1}')
    plt.savefig(Z_path)
    plt.close()

        

        
   