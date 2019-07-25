import numpy as np
import matplotlib.pyplot as plt
import pickle

#np.random.seed(42)
def hinton(matrix,path, max_weight=None, ax=None):

    #Draw Hinton diagram for visualizing a weight matrix.
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

data = 'ADNI/MMSE_scores'
scenario = 'complete'
noise = 'FA'
model = 'GFA'
m=40
directory = f'results/{data}/{noise}/{m}models/{scenario}/'
filepath = f'{directory}{model}_results.dictionary'
with open(filepath, 'rb') as parameters:

    model = pickle.load(parameters)

for i in range(0,len(model)):
    
    #Hinton diagrams for W1 and W2
    W_path = f'{directory}/estimated_Ws{i+1}.png'
    W1 = model[i].means_w[0]
    W2 = model[i].means_w[1]
    W = np.concatenate((W1.T,W2.T),axis=1)
    fig = plt.figure()
    fig.suptitle('Estimated Ws')
    hinton(W,W_path)

    #Hinton diagrams for alpha1 and alpha2
    a_path = f'{directory}/estimated_alphas{i+1}.png'
    a1 = np.reshape(model[i].E_alpha[0],(model[i].m,1))
    a2 = np.reshape(model[i].E_alpha[1],(model[i].m,1))
    a = np.concatenate((a1,a2),axis=1)
    fig = plt.figure()
    fig.suptitle('Estimated Alphas')
    hinton(-a.T,a_path)
    
    #plot lower bound
    L_path = f'{directory}/LB{i+1}.png'
    fig = plt.figure()
    fig.suptitle('Lower Bound')
    plt.plot(model[i].L[1:])
    plt.savefig(L_path)

    if data == 'simulations':
        #plot true latent variables
        Z_path = f'{directory}/true_Z{i+1}.png'
        x = np.linspace(0,99,100)
        numsub = model[i].Z.shape[1] 
        fig = plt.figure()
        fig.suptitle('True latent components')
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for j in range(1, numsub+1):
            ax = fig.add_subplot(4, 1, j)
            ax.scatter(x,model[i].Z[:,j-1])
        plt.savefig(Z_path)

        #plot true projections
        W_path = f'{directory}/true_Ws{i+1}.png'
        W1 = model[i].W[0]
        W2 = model[i].W[1]
        W = np.concatenate((W1.T,W2.T),axis=1)
        fig = plt.figure()
        fig.suptitle('True Ws')
        hinton(W,W_path)

        #plot estimated latent variables
        Z_path = f'{directory}/estimated_Z{i+1}.png'
        x = np.linspace(0,99,100)
        numsub = model[i].means_z.shape[1] 
        fig = plt.figure()
        fig.suptitle('Estimated latent components')
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for j in range(1, numsub+1):
            ax = fig.add_subplot(4, 1, j)
            ax.scatter(x,model[i].means_z[:,j-1])
        plt.savefig(Z_path)
        
        