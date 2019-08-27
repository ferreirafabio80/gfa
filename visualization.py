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


data = 'ADNI_joao/overall_scores_gender_brainclean' 
#data = 'simulations'
scenario = 'complete'
noise = 'PCA'
model = 'GFA'
m = 10
#m = 8
directory = f'results/{data}/{noise}/{m}models/{scenario}/'
filepath = f'{directory}{model}_results.dictionary'

# data
""" brain_data = pd.read_csv("results/ADNI/CT.csv")
brain_labels = pd.read_csv("results/ADNI/CT_labels.csv")
clinical_data = pd.read_csv("results/ADNI/clinical_overall.csv")
X_labels = brain_labels.AreaLabel.values
Y_labels = clinical_data.columns.T._values """
brain_labels = pd.read_csv("results/ADNI_joao/X_labels_clean.csv")
clinical_data = pd.read_csv("results/ADNI_joao/Y_labels.csv")
groups = pd.read_csv("results/ADNI_joao/groups.csv")
X_labels = brain_labels.Regions.values
Y_labels = clinical_data.clinical.values 
cohorts = groups.cohort.values
with open(filepath, 'rb') as parameters:

    model = pickle.load(parameters)

for i in range(0, len(model)):

    #latent spaces
    comp1 = model[i].means_z[:,1]
    comp2 = model[i].means_z[:,3]
    comp4 = model[i].means_z[:,4]
    Z_path = f'{directory}/latent_space{i+1}.png'
    colors = []
    for k in range(0,cohorts.shape[0]): 
        if cohorts[k] == 'CN':
            colors.append('b')
        elif cohorts[k] == 'AD':
            colors.append('r')
        else:
            colors.append('g')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(comp1, comp2, comp4, c=colors)
    #plt.title(f'Component 1vs2',fontsize=18)
    ax.set_xlabel('Component 2')
    ax.set_ylabel('Component 4')
    ax.set_zlabel('Component 5')
    #legend1 = ax.legend(*scatter.legend_elements(),loc='center left',title="Groups")
    #ax.add_artist(legend1)
    #plt.show()
    plt.savefig(Z_path)

    #Weights and total variance
    W1 = model[i].means_w[0]
    W2 = model[i].means_w[1]
    W = np.concatenate((W1, W2), axis=0)
    colMeans_W = np.mean(W ** 2, axis=0)
    var = colMeans_W * 100
    ind = np.argsort(-var)

    #Brain weights
    if i <= 2:
        numcomp = W.shape[1]
        for j in range(0, numcomp):
            W_path = f'{directory}/w_brain{i+1}_comp{j+1}.png'
            fig = plt.figure(figsize=(15, 10))
            colors = []
            #weight = np.sort(-W1[:, ind[j]])
            w_ind = np.argsort(-W1[:, ind[j]])
            if w_ind.shape[0] > 50:
                top = 20
                w_top = np.concatenate((w_ind[0:top], w_ind[w_ind.shape[0]-top:w_ind.shape[0]]))
                W1_sort = W1[w_top, ind[j]]
                labels = X_labels[w_top]
            else:    
                W1_sort = W1[w_ind, ind[j]]
                labels = X_labels[w_ind] 
            for k in range(0,W1_sort.shape[0]): 
                if W1_sort[k] > 0:
                    colors.append('r')
                else:
                    colors.append('b')
            x = range(W1_sort.shape[0])        
            plt.barh(labels[::-1], width=W1_sort[::-1], height=0.75,color=colors[::-1])
            plt.subplots_adjust(left=0.25,right=0.95)
            plt.title(f'Brain weights - Component {j+1}',fontsize=18)
            plt.xlim([-1, 1])
            plt.yticks(x, labels[::-1], size='small')
            
            #plt.show()    
            plt.savefig(W_path)
    
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

    # Clinical weights
    numsub = W.shape[1]
    W_path = f'{directory}/w_cli{i+1}.png'
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Clinical weights')
    fig.subplots_adjust(hspace=1.2, wspace=0.5)
    for j in range(0, numsub):
        ax = fig.add_subplot(numsub, 1, j+1)
        ax.title.set_text(f'Component {j+1}: total variance - {var[ind[j]]}%')
        ax.set_ylim([-1, 1])
        #weight = np.sort(-W1[:, ind[j]])
        w_ind = np.argsort(-W2[:, ind[j]])
        if w_ind.shape[0] > 50:
            top = 20
            w_top = np.concatenate((w_ind[0:top], w_ind[w_ind.shape[0]-top:w_ind.shape[0]]))
            W1_sort = W2[w_top, ind[j]]
            labels = Y_labels[w_top]
        else:    
            W2_sort = W2[w_ind, ind[j]]
            labels = Y_labels[w_ind]
        colors = []
        for k in range(0,W2_sort.shape[0]): 
            if W2_sort[k] > 0:
                colors.append('r')
            else:
                colors.append('b')
        ax.bar(labels, W2_sort,color=colors)
        
    plt.savefig(W_path)

    if data == 'simulations':
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

"""     N = X_labels.shape[0]
    bottom = 8
    max_height = 4

    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radii = max_height*W1[:,0]
    width = (2*np.pi) / N

    ax = plt.subplot(111, polar=True)
    bars = ax.bar(theta, radii, tick_label=X_labels, width=width, bottom=bottom)
    ax.spines['polar'].set_visible(False)
    ax.grid(False)
    ax.set_yticklabels([])
    ax.tick_params(axis ='x', rotation=90, direction='out',
            width = 10) 

    # Use custom colors and opacity
    for r, bar in zip(radii, bars):
        bar.set_facecolor(plt.cm.jet(r / 10.))
        bar.set_alpha(0.8)        

    plt.show() """
   