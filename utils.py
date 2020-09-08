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
        sigmaZ = np.identity(self.model.k)
        for i in range(train[0].size): 
            if 'spherical' in noise:
                sigmaZ = sigmaZ + self.model.E_tau[train[0][i]] * self.model.E_WW[train[0][i]]
            else:
                for j in range(self.model.d[train[0,0]]):
                    w = np.reshape(self.model.means_w[train[0,i]][j,:], (1,self.model.k))
                    ww = self.model.sigma_w[train[0,i]][:,:,j] + np.dot(w.T, w) 
                    sigmaZ = sigmaZ + self.model.E_tau[train[0,i]][0,j] * ww

        # Estimate the latent variables       
        w, v = np.linalg.eig(sigmaZ)
        sigmaZ = np.dot(v * np.outer(np.ones((1,self.model.k)), 1/w), v.T)
        meanZ = np.zeros((N,self.model.k))
        for i in range(train[0].size):
            if 'spherical' in noise: 
                meanZ = meanZ + np.dot(self.X[train[0][i]], self.model.means_w[train[0][i]]) * self.model.E_tau[train[0][i]]
            else: 
                for j in range(self.model.d[train[0,0]]):
                    w = np.reshape(self.model.means_w[train[0,i]][j,:], (1,self.model.k)) 
                    x = np.reshape(self.X[train[0,i]][:,j], (N,1)) 
                    meanZ = meanZ + np.dot(x, w) * self.model.E_tau[train[0,i]][0,j]         
        meanZ = np.dot(meanZ, sigmaZ)
        
        X_pred = np.dot(meanZ, self.model.means_w[pred[0]].T)             
        return X_pred

    def PredictMissing(self, missTrain=False, missRows=False):
        train = np.array(np.where(self.view == 0))
        pred = np.array(np.where(self.view == 1))   
        N = self.X[0].shape[0] #number of samples

        if missTrain:
            #Estimate the covariance of the latent variables
            sigmaZ = np.zeros((self.model.k,self.model.k,N))
            for n in range(0, N):
                sigmaZ[:,:,n] = np.identity(self.model.k)
            for n in range(0, N):
                S = np.zeros((self.model.k,self.model.k))
                for i in range(0, train[0].size):
                    for j in range(self.model.d[train[0,0]]):
                        if ~np.isnan(self.X[train[0,i]][n,j]):
                            w = np.reshape(self.model.means_w[train[0,i]][j,:], (1,self.model.k))
                            ww = self.model.sigma_w[train[0,i]][:,:,j] + np.dot(w.T, w)
                            S += self.model.E_tau[train[0,i]][0,j] * ww
                sigmaZ[:,:,n] = sigmaZ[:,:,n] + S

            #Estimate expectation of latent variables       
            meanZ = np.zeros((N,self.model.k))       
            for n in range(0, self.X[train[0,i]].shape[0]):
                S = np.zeros((1,self.model.k))
                w, v = np.linalg.eig(sigmaZ[:,:,n])
                sigmaZ[:,:,n] = np.dot(v * np.outer(np.ones((1,self.model.k)), 1/w), v.T) 
                for i in range(0, train.size):
                    for j in range(0, self.X[train[0,i]].shape[1]):
                        if ~np.isnan(self.X[train[0,i]][n,j]):
                            w = np.reshape(self.model.means_w[train[0,i]][j,:], (1,self.model.k)) 
                            x = self.X[train[0,i]][n,j]
                            S += x * w * self.model.E_tau[train[0,i]][0,j]
                meanZ[n,:] = np.dot(S, sigmaZ[:,:,n])
        else:
            #Estimate the covariance of the latent variables
            sigmaZ = np.identity(self.model.k)
            for i in range(train[0].size):
                for j in range(self.model.d[train[0,0]]):
                        w = np.reshape(self.model.means_w[train[0,i]][j,:], (1,self.model.k))
                        ww = self.model.sigma_w[train[0,i]][:,:,j] + np.dot(w.T, w) 
                        sigmaZ = sigmaZ + self.model.E_tau[train[0,i]][0,j] * ww                        
            #Estimate expectation of latent variables
            w, v = np.linalg.eig(sigmaZ)
            sigmaZ = np.dot(v * np.outer(np.ones((1,self.model.k)), 1/w), v.T)
            meanZ = np.zeros((N,self.model.k))
            for i in range(train[0].size):
                for j in range(self.model.d[train[0,0]]):
                    w = np.reshape(self.model.means_w[train[0,i]][j,:], (1,self.model.k)) 
                    x = np.reshape(self.X[train[0,i]][:,j], (N,1)) 
                    meanZ = meanZ + np.dot(x, w) * self.model.E_tau[train[0,i]][0,j]         
            meanZ = np.dot(meanZ, sigmaZ)
                            
        X_pred = [[] for _ in range(self.model.s)]
        for i in range(0, pred.size):
            X_pred[pred[0,i]] = np.zeros((N, self.model.d[pred[0,i]]))
            if missRows:
                for n in range(0, self.X[pred[0,i]].shape[0]):
                    if np.isnan(self.X[pred[0,i]][n,:]).any():
                            X_pred[pred[0,i]][n,:] = np.dot(meanZ[n,:], self.model.means_w[pred[0,i]].T)
            else:    
                for n in range(0, self.X[pred[0,i]].shape[0]):
                    for j in range(0, self.X[pred[0,i]].shape[1]):
                        if np.isnan(self.X[pred[0,i]][n,j]):
                            X_pred[pred[0,i]][n,j] = np.dot(meanZ[n,:], self.model.means_w[pred[0,i]][j,:].T)          

        return X_pred