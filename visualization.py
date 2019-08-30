import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from scipy import io
from mpl_toolkits.mplot3d import Axes3D


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

def plot_wcli(var, w_cli, l_cli, path_cli):
    ind = np.argsort(-var)
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Clinical weights',fontsize=20)
    fig.subplots_adjust(hspace=1.3, wspace=0.5)
    comp = w_cli.shape[1]
    for j in range(0, comp):
        ax = fig.add_subplot(comp, 1, j+1)
        ax.title.set_text(f'Component {j+1}: total variance - {var[ind[j]]}%')
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
            w_sort = w[w_ind]
            l_sort = l_cli[w_ind]
       
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

def plot_Z(comp1, comp2, path_z):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    for N,k in zip(range(comp1.shape[0]),cohorts): 
        if k == 'AD':
            ax.scatter(comp1[N], comp2[N], c='red', s = 50, alpha=0.6, edgecolors='none')
        elif k == 'CN':
            ax.scatter(comp1[N], comp2[N], c='green', s = 50, alpha=0.6, edgecolors='none')
        else:
            ax.scatter(comp1[N], comp2[N], c='orange', s = 50, alpha=0.6, edgecolors='none')
    plt.title(f'Latent space',fontsize=18)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend(labels=('CN','AD','MCI'),loc='upper right',title="Groups")
    plt.savefig(Z_path)
    plt.close()

data = 'ADNI' #simulations
flag = '_joao/overall_scores_gender_brainclean'
noise = 'FA' 
scenario = 'missing20'
model = 'GFA'
m = 15
if model == 'GFA':
    directory = f'results/{data}{flag}/{noise}/{m}models/{scenario}/'        
    filepath = f'{directory}{model}_results.dictionary'
else:
    directory = f'results/{data}{flag}/{model}/'

if data=='ADNI':
    #Labels
    brain_labels = pd.read_csv("results/ADNI_joao/X_labels_clean.csv")
    clinical_labels = pd.read_csv("results/ADNI_joao/Y_labels.csv")
    groups = pd.read_csv("results/ADNI_joao/groups.csv")
    X_labels = brain_labels.Regions.values
    Y_labels = clinical_labels.clinical.values 
    cohorts = groups.cohort.values

    if model == 'GFA':
        #Load file
        with open(filepath, 'rb') as parameters:

            model = pickle.load(parameters)   
        
        #Plot weights, ELBO, alphas and latent spaces for each random init
        for i in range(0, len(model)):
            #Weights and total variance
            W1 = model[i].means_w[0]
            W2 = model[i].means_w[1]
            W = np.concatenate((W1, W2), axis=0)
            colMeans_W = np.mean(W ** 2, axis=0)
            var = colMeans_W * 100
            ind = np.argsort(-var)
            numcomp = W.shape[1]

            #Brain weights
            for j in range(0, numcomp):  
                brain_path = f'{directory}/w_brain{i+1}_comp{j+1}.png'
                plot_wbrain(j, W1[:,ind[j]], X_labels, brain_path)
        
            #Clinical weights
            cli_path = f'{directory}/w_cli{i+1}.png'
            plot_wcli(var, W2, Y_labels, cli_path)

            #Latent spaces
            comp1 = model[i].means_z[:,0]
            comp2 = model[i].means_z[:,1]
            Z_path = f'{directory}/latent_space{i+1}.png'
            plot_Z(comp1, comp2, Z_path)

            #Hinton diagrams for alpha1 and alpha2
            a_path = f'{directory}/estimated_alphas{i+1}.png'
            a1 = np.reshape(model[i].E_alpha[0], (model[i].m, 1))
            a2 = np.reshape(model[i].E_alpha[1], (model[i].m, 1))
            a = np.concatenate((a1, a2), axis=1)
            plt.figure()
            plt.title('Estimated Alphas')
            hinton(-a.T, a_path)
            plt.close()

            #Plot lower bound
            L_path = f'{directory}/LB{i+1}.png'
            plt.figure()
            plt.title('Lower Bound')
            plt.plot(model[i].L[1:])
            plt.savefig(L_path)
            plt.close()
            
    else:
        if model=='CCA':
            fw = 'permutation' 
            #Clinical weights
            filepath = f'{directory}/wcli_{fw}.mat'
            cli_path = f'{directory}/wcli_{fw}.png'
            weights = io.loadmat(filepath)
            w = weights['w2']
            var = np.array((30,20,10))
            plot_wcli(var, w, Y_labels, cli_path)

            #Brain weights
            filepath = f'{directory}/wbrain_{fw}.mat'
            weights = io.loadmat(filepath)
            w = weights['w1']
            var = np.array((30,20,10))
            ind = np.argsort(-var)
            numcomp = w.shape[1]
            for i in range(0, numcomp):  
                brain_path = f'{directory}/wbrain_{fw}{i+1}.png'
                plot_wbrain(i, w[:,ind[i]], X_labels, brain_path)
        
        elif model=='SCCA':
            #Clinical weights
            filepath = f'{directory}/wcli.mat'
            cli_path = f'{directory}/wcli.png'
            weights = io.loadmat(filepath)
            w = weights['v']
            var = np.array((30,20,10))
            plot_wcli(var, w, Y_labels, cli_path)

            #Brain weights
            filepath = f'{directory}/wbrain.mat'
            weights = io.loadmat(filepath)
            w = weights['u']
            var = np.array((30,20,10))
            ind = np.argsort(-var)
            numcomp = w.shape[1]
            for i in range(0, numcomp):  
                brain_path = f'{directory}/wbrain{i+1}.png'
                plot_wbrain(i, w[:,ind[i]], X_labels, brain_path)

elif data == 'simulations':
    # Hinton diagrams for W1 and W2
    W1 = model[i].means_w[0]
    W2 = model[i].means_w[1]
    W = np.concatenate((W1, W2), axis=0)
    W_path = f'{directory}/estimated_Ws{i+1}.png'
    fig = plt.figure()
    fig.suptitle('Estimated Ws')
    hinton(W, W_path)

    # Hinton diagrams for alpha1 and alpha2
    a_path = f'{directory}/estimated_alphas{i+1}.png'
    a1 = np.reshape(model[i].E_alpha[0], (model[i].m, 1))
    a2 = np.reshape(model[i].E_alpha[1], (model[i].m, 1))
    a = np.concatenate((a1, a2), axis=1)
    fig = plt.figure()
    fig.suptitle('Estimated Alphas')
    hinton(-a.T, a_path)

    # plot lower bound
    L_path = f'{directory}/LB{i+1}.png'
    fig = plt.figure()
    fig.suptitle('Lower Bound')
    plt.plot(model[i].L[1:])
    plt.savefig(L_path)

    # plot true latent variables
    Z_path = f'{directory}/true_Z{i+1}.png'
    x = np.linspace(0, 99, 100)
    numsub = model[i].Z.shape[1]
    fig = plt.figure()
    fig.suptitle('True latent components')
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for j in range(1, numsub+1):
        ax = fig.add_subplot(4, 1, j)
        ax.scatter(x, model[i].Z[:, j-1])
    plt.savefig(Z_path)

    # plot true projections
    W_path = f'{directory}/true_Ws{i+1}.png'
    W1 = model[i].W[0]
    W2 = model[i].W[1]
    W = np.concatenate((W1.T, W2.T), axis=1)
    fig = plt.figure()
    fig.suptitle('True Ws')
    hinton(W, W_path)

    # plot estimated latent variables
    Z_path = f'{directory}/estimated_Z{i+1}.png'
    x = np.linspace(0, 99, 100)
    numsub = model[i].means_z.shape[1]
    fig = plt.figure()
    fig.suptitle('Estimated latent components')
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for j in range(1, numsub+1):
        ax = fig.add_subplot(4, 1, j)
        ax.scatter(x, model[i].means_z[:, j-1])
    plt.savefig(Z_path)
   