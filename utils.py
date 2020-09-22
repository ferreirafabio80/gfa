"""Save the results of the experiments on synthetic data"""

#Author: Fabio S. Ferreira (fabio.ferreira.16@ucl.ac.uk)
#Date: 17 September 2020

import numpy as np

class GFAtools(object):
    def __init__(self, X, model):
        self.X = X
        self.model = model

    def PredictGroups(self, obs_groups, noise):

        """ 
        Predict non-observed groups from observed ones.

        Parameters
        ----------
        obs_groups : array-like 
            Info about the groups to be predicted. 1 represents 
            observed groups. 0 represents non-observed groups.

        noise : str
            Noise assumption.  

        Returns
        -------
        X_pred : list
            List of arrays containing the predicted group(s).
        
        """
        train = np.where(obs_groups == 1)[0] #observed groups
        pred = np.where(obs_groups == 0)[0] #non-observed groups   
        N = self.X[0].shape[0] #number of samples
        
        # Estimate the covariance of the latent variables
        sigmaZ = np.identity(self.model.k)
        for i in range(train.size): 
            if 'spherical' in noise:
                sigmaZ = sigmaZ + self.model.E_tau[train[i]] * self.model.E_WW[train[i]]
            else:
                for j in range(self.model.d[train[i]]):
                    w = np.reshape(self.model.means_w[train[i]][j,:], (1,self.model.k))
                    ww = self.model.sigma_w[train[i]][:,:,j] + np.dot(w.T, w) 
                    sigmaZ = sigmaZ + self.model.E_tau[train[i]][0,j] * ww
        
        # Estimate expectations of the latent variables       
        w, v = np.linalg.eig(sigmaZ)
        sigmaZ = np.dot(v * np.outer(np.ones((1,self.model.k)), 1/w), v.T)
        meanZ = np.zeros((N,self.model.k))
        for i in range(train.size):
            if 'spherical' in noise: 
                meanZ = meanZ + np.dot(self.X[train[i]], self.model.means_w[train[i]]) * self.model.E_tau[train[i]]
            else: 
                for j in range(self.model.d[train[i]]):
                    w = np.reshape(self.model.means_w[train[i]][j,:], (1,self.model.k)) 
                    x = np.reshape(self.X[train[i]][:,j], (N,1)) 
                    meanZ = meanZ + np.dot(x, w) * self.model.E_tau[train[i]][0,j]         
        meanZ = np.dot(meanZ, sigmaZ)
        
        #Predict non-observed groups  
        X_pred = [[] for _ in range(pred.size)]
        for p in range(pred.size):
            X_pred[p] = np.dot(meanZ, self.model.means_w[pred[0]].T)             
        return X_pred

    def PredictMissing(self, infoMiss):

        """ 
        Predict missing values.

        Parameters
        ----------
        infoMiss : dict 
            Parameters to generate data with missing values.  

        Returns
        -------
        X_pred : list
            List of arrays with predicted missing values.
        
        """
        pred = np.array(infoMiss['group']) - 1 #group with missing values   
        N = self.X[0].shape[0] #number of samples
        if len(infoMiss['group']) > 1:
            #Estimate the covariance of the latent variables
            sigmaZ = np.zeros((self.model.k,self.model.k,N))
            for n in range(N):
                S = np.identity(self.model.k)
                for i in range(self.model.s):
                    for j in range(self.model.d[i]):
                        if ~np.isnan(self.X[i][n,j]):
                            w = np.reshape(self.model.means_w[i][j,:], (1,self.model.k))
                            ww = self.model.sigma_w[i][:,:,j] + np.dot(w.T, w)
                            S += self.model.E_tau[i][0,j] * ww
                sigmaZ[:,:,n] = S

            #Estimate expectations of the latent variables       
            meanZ = np.zeros((N,self.model.k))       
            for n in range(N):
                S = np.zeros((1,self.model.k))
                w, v = np.linalg.eig(sigmaZ[:,:,n])
                sigmaZ[:,:,n] = np.dot(v * np.outer(np.ones((1,self.model.k)), 1/w), v.T) 
                for i in range(self.model.s):
                    for j in range(self.X[i].shape[1]):
                        if ~np.isnan(self.X[i][n,j]):
                            w = np.reshape(self.model.means_w[i][j,:], (1,self.model.k)) 
                            x = self.X[i][n,j]
                            S += x * w * self.model.E_tau[i][0,j]
                meanZ[n,:] = np.dot(S, sigmaZ[:,:,n])
        else:
            g_nomiss = np.ones((1,self.model.s))
            g_nomiss[0,pred] = 0
            train = np.where(g_nomiss[0,:] == 1)[0]
            #Estimate the covariance of the latent variables
            sigmaZ = np.identity(self.model.k)
            for i in range(train.size):
                for j in range(self.model.d[train[i]]):
                    w = np.reshape(self.model.means_w[train[i]][j,:], (1,self.model.k))
                    ww = self.model.sigma_w[train[i]][:,:,j] + np.dot(w.T, w) 
                    sigmaZ = sigmaZ + self.model.E_tau[train[i]][0,j] * ww                        
            #Estimate expectation of latent variables
            w, v = np.linalg.eig(sigmaZ)
            sigmaZ = np.dot(v * np.outer(np.ones((1,self.model.k)), 1/w), v.T)
            meanZ = np.zeros((N,self.model.k))
            for i in range(train.size):
                for j in range(self.model.d[train[i]]):
                    w = np.reshape(self.model.means_w[train[i]][j,:], (1,self.model.k)) 
                    x = np.reshape(self.X[train[i]][:,j], (N,1)) 
                    meanZ = meanZ + np.dot(x, w) * self.model.E_tau[train[i]][0,j]         
            meanZ = np.dot(meanZ, sigmaZ)
        
        #Predict missing values only                    
        X_pred = [[] for _ in range(pred.size)]
        for i in range(pred.size):
            X_pred[i] = np.zeros((N, self.model.d[pred[i]]))
            if 'rows' in infoMiss['type'][i]:
                for n in range(N):
                    if np.isnan(self.X[pred[i]][n,:]).any():
                            X_pred[i][n,:] = np.dot(meanZ[n,:], self.model.means_w[pred[i]].T)
            else:    
                for n in range(N):
                    for j in range(0, self.X[pred[i]].shape[1]):
                        if np.isnan(self.X[pred[i]][n,j]):
                            X_pred[i][n,j] = np.dot(meanZ[n,:], self.model.means_w[pred[i]][j,:].T)          
        return X_pred