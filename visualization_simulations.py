import numpy as np
import matplotlib.pyplot as plt
import pickle


def hinton(matrix, max_weight=None, ax=None):

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
    plt.show()

with open('results/simulations/BIBFA_completePCA.dictionary', 'rb') as parameters:
 
    # Step 3
    model = pickle.load(parameters)

#Hinton diagrams for estimated projections
W1 = model.means_w[0]
W2 = model.means_w[1]
W = np.concatenate((W1.T,W2.T),axis=1)
hinton(W)

#Hinton diagrams for alpha1 and alpha2
a1 = np.reshape(model.E_alpha[0],(K,1))
a2 = np.reshape(model.E_alpha[1],(K,1))
a = np.concatenate((a1,a2),axis=1)
hinton(-a.T)

print("Estimated variances:", model.E_tau)
print("Estimated alphas:", model.E_alpha)

#plot lower bound
plt.plot(L[1:])
plt.show()

#plot true latent variables
x = np.linspace(0,99,100)
f, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, sharex='col', sharey='row')
f.suptitle('True latent components')
ax1.scatter(x,Z_train[:,0])
ax2.scatter(x,Z_train[:,1])
ax3.scatter(x,Z_train[:,2])
ax4.scatter(x,Z_train[:,3])
plt.show()

#plot estimated latent variables
x = np.linspace(0,199,200)
f, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, sharex='col', sharey='row')
f.suptitle('Estimated latent components')
ax1.scatter(x,model.means_z[:,0])
ax2.scatter(x,model.means_z[:,1])
ax3.scatter(x,model.means_z[:,2])
ax4.scatter(x,model.means_z[:,3])
plt.show()
