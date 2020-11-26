import numpy as np
import matplotlib.pyplot as plt
import hdf5storage
from scipy import io

exp_dir = '/Users/fabioferreira/Desktop/repos/GFA/results/CCA_comparison/CCA/incomplete2'
data_dir = f'{exp_dir}/data'
res_dir = f'{exp_dir}/framework/cca_permutation_TEST/res'

X = hdf5storage.loadmat(f'{data_dir}/X.mat')
Y = hdf5storage.loadmat(f'{data_dir}/Y.mat')
X = X['X']; Y=Y['Y']
medians_X = np.nanmedian(X,axis=0)
for j in range(X.shape[1]):
    X[np.isnan(X[:,j]),j] = medians_X[j]
medians_Y = np.nanmedian(Y,axis=0)
for j in range(Y.shape[1]):
    Y[np.isnan(Y[:,j]),j] = medians_Y[j]

levels=2
Z = np.zeros(shape=(X.shape[0],levels))
for i in range(levels):
    lev_dir = f'{res_dir}/level{i+1}'
    mat = hdf5storage.loadmat(f'{lev_dir}/model_1.mat')
    wx = mat['wX']
    wy = mat['wY']
    Z[:,i] = ((np.dot(X,wx.T)+ np.dot(Y,wy.T))/2)[:,0]

x = np.linspace(0, Z.shape[0], Z.shape[0])
ncomp = Z.shape[1]
fig = plt.figure(figsize=(8, 6))
fig.subplots_adjust(hspace=0.1, wspace=0.1)
Z_path = f'{data_dir}/Z_est.png'
for j in range(ncomp):
    ax = fig.add_subplot(ncomp, 1, j+1)    
    ax.scatter(x, Z[:, j], s=4)
    ax.set_xticks([])
    ax.set_yticks([])       
plt.savefig(Z_path)
plt.close()
