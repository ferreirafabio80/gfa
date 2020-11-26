#Author: Fabio S. Ferreira (fabio.ferreira.16@ucl.ac.uk)
#Date: 17 September 2020

import numpy as np

class GFAtools(object):
    def __init__(self, X, model):
        self.X = X
        self.model = model

    def PredictDSources(self, obs_ds, noise):

        """ 
        Predict non-observed data sources from observed ones.

        Parameters
        ----------
        obs_ds : array-like 
            Info about the data sources to be predicted. 1 
            represents observed data sources. 0 represents 
            non-observed data sources.

        noise : str
            Noise assumption.  

        Returns
        -------
        X_pred : list
            List of arrays containing the predicted data source(s).
        
        """
        train = np.where(obs_ds == 1)[0] #observed data sources
        pred = np.where(obs_ds == 0)[0] #non-observed data sources   
        N = self.X[0].shape[0] #number of samples
        
        # Estimate the covariance of the latent variables
        sigmaZ = np.identity(self.model.k)
        for i in range(train.size): 
            if 'spherical' in noise:
                sigmaZ = sigmaZ + self.model.E_tau[0,train[i]] * self.model.E_WW[train[i]]
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
                meanZ = meanZ + np.dot(self.X[train[i]], self.model.means_w[train[i]]) * self.model.E_tau[0,train[i]]
            else: 
                for j in range(self.model.d[train[i]]):
                    w = np.reshape(self.model.means_w[train[i]][j,:], (1,self.model.k)) 
                    x = np.reshape(self.X[train[i]][:,j], (N,1)) 
                    meanZ = meanZ + np.dot(x, w) * self.model.E_tau[train[i]][0,j]         
        meanZ = np.dot(meanZ, sigmaZ)
        
        #Predict non-observed data sources  
        X_pred = [[] for _ in range(pred.size)]
        for p in range(pred.size):
            X_pred[p] = np.dot(meanZ, self.model.means_w[pred[p]].T)             
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
        pred = infoMiss['ds'] #data source with missing values   
        train = np.arange(self.model.s)
        N = self.X[0].shape[0] #number of samples
        X_pred = [[] for _ in range(len(pred))]
        for p in range(len(pred)):
            t_tmp = np.delete(train,pred[p]-1)
            #Estimate the covariance of the latent variables
            sigmaZ = np.zeros((self.model.k,self.model.k,N))
            for n in range(N):
                S = np.identity(self.model.k)
                for t in t_tmp:
                    for j in range(self.model.d[t]):
                        if ~np.isnan(self.X[t][n,j]):
                            w = np.reshape(self.model.means_w[t][j,:], (1,self.model.k))
                            ww = self.model.sigma_w[t][:,:,j] + np.dot(w.T, w)
                            S += self.model.E_tau[t][0,j] * ww
                sigmaZ[:,:,n] = S

            #Estimate expectations of the latent variables       
            meanZ = np.zeros((N,self.model.k))       
            for n in range(N):
                S = np.zeros((1,self.model.k))
                w, v = np.linalg.eig(sigmaZ[:,:,n])
                sigmaZ[:,:,n] = np.dot(v * np.outer(np.ones((1,self.model.k)), 1/w), v.T) 
                for t in t_tmp:
                    for j in range(self.model.d[t]):
                        if ~np.isnan(self.X[t][n,j]):
                            w = np.reshape(self.model.means_w[t][j,:], (1,self.model.k)) 
                            x = self.X[t][n,j]
                            S += x * w * self.model.E_tau[t][0,j]
                meanZ[n,:] = np.dot(S, sigmaZ[:,:,n])

            #Predict missing values only
            p_ind = pred[p]-1
            X_pred[p] = np.empty((N, self.model.d[p_ind]))
            X_pred[p][:,:] = np.NaN
            if 'rows' in infoMiss['type'][p]:
                for n in range(N):
                    if np.isnan(self.X[p_ind][n,:]).any():
                        X_pred[p][n,:] = np.dot(meanZ[n,:], self.model.means_w[p_ind].T)
            else:    
                for n in range(N):
                    for j in range(0, self.model.d[p_ind]):
                        if np.isnan(self.X[p_ind][n,j]):
                            X_pred[p][n,j] = np.dot(meanZ[n,:], self.model.means_w[p_ind][j,:].T)    
        return X_pred