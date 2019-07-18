import numpy as np
import matplotlib.pyplot as plt
import pickle

np.random.seed(42)
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

with open('BCCAdiag_missing20_sample300.dictionary', 'rb') as parameters:
 
    # Step 3
    BCCA = pickle.load(parameters)

#Hinton diagrams for W1 and W2
W1 = BCCA.means_w[0]
W2 = BCCA.means_w[1]
W = np.concatenate((W1,W2),axis=0)
hinton(W)

#Hinton diagrams for alpha1 and alpha2
a1 = np.reshape(BCCA.E_alpha[0],(BCCA.m,1))
a2 = np.reshape(BCCA.E_alpha[1],(BCCA.m,1))
a = np.concatenate((a1,a2),axis=1)
hinton(-a.T)

print("Estimated variances:", BCCA.E_tau[0])
print("Estimated variances:", BCCA.E_tau[1])

#plot lower bound
plt.plot(BCCA.L[1:])
plt.show()

#plot estimated latent variables
x = np.linspace(0,99,100)
f, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, sharex='col', sharey='row')
f.suptitle('Estimated latent components')
ax1.scatter(x,BCCA.means_z[:,0])
ax2.scatter(x,BCCA.means_z[:,1])
ax3.scatter(x,BCCA.means_z[:,2])
ax4.scatter(x,BCCA.means_z[:,3])
plt.show()