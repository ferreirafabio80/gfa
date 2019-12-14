import numpy as np
import numpy.ma as ma
import numpy.linalg as LA
import math

class GFAtools(object):
    def __init__(self, X, model, view):
        self.X = X
        self.model = model
        self.view = view

    def PredictView(self, noise):
        train = np.array(np.where(self.view == 1))
        pred = np.array(np.where(self.view == 0)) 
        if not pred[0].size:
            pred = np.array(range(0,self.model.s))
        else:
            pred = pred[0]    
        N = self.X[0].shape[0] #number of samples

        # Estimate the covariance of the latent variables
        sigmaZ = np.identity(self.model.m)
        for i in range(train[0].shape[0]): 
            if 'PCA' in noise:
                sigmaZ = sigmaZ + self.model.E_tau[train[0][i]] * self.model.E_WW[train[0][i]]
            else:
                for j in range(self.model.d[train[0,0]]):
                    w = np.reshape(self.model.means_w[train[0,i]][j,:], (1,self.model.m))
                    ww = self.model.sigma_w[train[0,i]][:,:,j] + np.dot(w.T, w) 
                    sigmaZ = sigmaZ + self.model.E_tau[train[0,i]][0,j] * ww

        # Estimate the latent variables       
        w, v = np.linalg.eig(sigmaZ)
        sigmaZ = np.dot(v * np.outer(np.ones((1,self.model.m)), 1/w), v.T)
        meanZ = np.zeros((N,self.model.m))
        for i in range(train[0].shape[0]):
            if 'PCA' in noise: 
                meanZ = meanZ + np.dot(self.X[train[0][i]], self.model.means_w[train[0][i]]) * self.model.E_tau[train[0][i]]
            else: 
                for j in range(self.model.d[train[0,0]]):
                    w = np.reshape(self.model.means_w[train[0,i]][j,:], (1,self.model.m)) 
                    x = np.reshape(self.X[train[0,i]][:,j], (N,1)) 
                    meanZ = meanZ + np.dot(x, w) * self.model.E_tau[train[0,i]][0,j]         
        meanZ = np.dot(meanZ, sigmaZ)

        # Add a tiny amount of noise on top of the latent variables,
        # to supress possible artificial structure in components that
        # have effectively been turned off
        Noise = 1e-05
        meanZ = meanZ + Noise * \
            np.dot(np.reshape(np.random.normal(
                0, 1, N * self.model.m),(N, self.model.m)), np.linalg.cholesky(sigmaZ)) 

        X_pred = np.dot(meanZ, self.model.means_w[pred[0]].T)          
        if 'PCA' in noise:
            sigma_pred = np.identity(self.model.d[pred[0]]) * 1/np.sqrt(self.model.E_tau[pred[0]])
        else:
            sigma_pred = np.diag(1/np.sqrt(self.model.E_tau[pred[0]])[0])        

        return X_pred, sigma_pred

    def PredictMissing(self):
        train = np.array(np.where(self.view == 1))
        pred = np.array(np.where(self.view == 0))   
        N = self.X[0].shape[0] #number of samples

        # Estimate the covariance of the latent variables
        sigmaZ = np.identity(self.model.m)
        for i in range(0, train[0].shape[0]):
            for j in range(self.model.d[train[0,0]]):
                w = np.reshape(self.model.means_w[train[0,i]][j,:], (1,self.model.m))
                ww = self.model.sigma_w[train[0,i]][:,:,j] + np.dot(w.T, w) 
                sigmaZ = sigmaZ + self.model.E_tau[train[0,i]][0,j] * ww

        # Estimate the latent variables       
        w, v = np.linalg.eig(sigmaZ)
        sigmaZ = np.dot(v * np.outer(np.ones((1,self.model.m)), 1/w), v.T)
        meanZ = np.zeros((N,self.model.m))
        for i in range(0, train.shape[0]):
            for j in range(self.model.d[train[0,0]]):
                w = np.reshape(self.model.means_w[train[0,i]][j,:], (1,self.model.m)) 
                x = np.reshape(self.X[train[0,i]][:,j], (N,1)) 
                meanZ = meanZ + np.dot(x, w) * self.model.E_tau[train[0,i]][0,j] 
        meanZ = np.dot(meanZ, sigmaZ)

        # Add a tiny amount of noise on top of the latent variables,
        # to supress possible artificial structure in components that
        # have effectively been turned off
        noise = 1e-05
        meanZ = meanZ + noise * \
            np.dot(np.reshape(np.random.normal(
                0, 1, N * self.model.m),(N, self.model.m)), np.linalg.cholesky(sigmaZ)) 

        X_pred = [[] for _ in range(self.model.s)]
        for i in range(0, pred.shape[0]):
            X_pred[pred[0,i]] = np.zeros((N, self.model.d[pred[0,i]]))
            for n in range(0, self.X[pred[0,i]].shape[0]):
                for j in range(0, self.X[pred[0,i]].shape[1]):
                    if np.isnan(self.X[pred[0,i]][n,j]):
                        X_pred[pred[0,i]][n,j] = np.dot(meanZ[n,:], self.model.means_w[pred[0,i]][j,:].T)          

        return X_pred